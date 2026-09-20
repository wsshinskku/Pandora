"""Strict JSON-lines policy boundary for an external RAN process.

The protocol is a local integration boundary, not an E2AP/E2SM implementation.
Applied-action acknowledgements are recorded separately from projected actions.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .actions import ActionSpace
from .calibration import RiskCalibrator
from .config import Config, ContractConfig, LearningConfig, SimulationConfig
from .contracts import fallback_contract
from .controller import PandoraController
from .data import Episode
from .learning import ImpactModel
from .synthesis import ContractSynthesizer


def load_controller(checkpoint, calibration_path, slices, seed=0):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw = saved["config"]
    config = Config(
        simulation=SimulationConfig(**raw["simulation"]),
        contract=ContractConfig(**raw["contract"]),
        learning=LearningConfig(**raw["learning"]),
        **{k: v for k, v in raw.items() if k not in {"simulation", "contract", "learning"}},
    ).validate()
    model = ImpactModel(saved["input_dim"], config.learning.personalize)
    model.load_state_dict(saved["state_dict"])
    calibration = Episode.load(calibration_path)
    calibrator = RiskCalibrator(config.contract, config.learning.calibrate).fit(model, calibration)
    space = ActionSpace(config.simulation.users, config.simulation.cells)
    slices = np.asarray(slices, dtype=int)
    if slices.shape != (space.users,) or np.any((slices < 0) | (slices > 2)):
        raise ValueError("One slice ID (0, 1 or 2) is required for each UE")
    H, limits = space.coupling_template(slices)
    if not config.learning.coupling:
        H, limits = np.empty((0, space.dim)), np.empty(0)
    fallback = fallback_contract(space, calibration.executed, H, limits)
    synthesis = ContractSynthesizer(
        space, config.contract, model, calibrator, H, limits, fallback, seed
    )
    return PandoraController(synthesis)


class JsonlSession:
    def __init__(self, controller, audit_path):
        self.controller = controller
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        if self.audit_path.exists():
            raise ValueError("Use a new audit path for each session; resume is not implicit")
        self.slot, self.pending, self.finished = 0, None, False
        self.last_request, self.last_response = None, None

    def _feedback_record(self, feedback):
        if self.pending is None:
            if feedback is not None:
                raise ValueError("Unexpected feedback before first control decision")
            return None
        if not isinstance(feedback, dict) or feedback.get("slot") != self.pending["slot"]:
            raise ValueError("Feedback for the previous slot is required")
        applied = np.asarray(feedback.get("executed"), dtype=float)
        kpis = np.asarray(feedback.get("kpis"), dtype=float)
        if (
            not self.controller.synthesizer.space.valid(applied)
            or kpis.shape != (5,)
            or not np.isfinite(kpis).all()
        ):
            raise ValueError("Invalid executed action or KPI feedback")
        if np.any(kpis[:2] < 0) or np.any(kpis[2:] < 0) or np.any(kpis[2:] > 1):
            raise ValueError("KPIs must use Mbps, ms and ratios in [0,1]")
        return {**self.pending, "executed": applied.tolist(), "kpis": kpis.tolist()}

    def handle(self, request):
        canonical = json.dumps(request, sort_keys=True, allow_nan=False)
        if canonical == self.last_request:
            return self.last_response
        if self.finished or request.get("version") != 1:
            raise ValueError("Session finished or unsupported protocol version")
        if request.get("slot") != self.slot:
            raise ValueError(f"Expected slot {self.slot}; duplicate requests must be identical")
        record = self._feedback_record(request.get("feedback"))
        kind = request.get("type", "step")
        if kind == "finish":
            if self.pending is None:
                raise ValueError("Cannot finish an empty session")
            response = {"version": 1, "slot": self.slot, "finished": True}
            self.finished = True
        elif kind == "step":
            observation = np.asarray(request.get("observation"), dtype=float)
            proposal = np.asarray(request.get("proposal"), dtype=float)
            s = self.controller.synthesizer
            obs_dim = s.model.input_dim - s.space.dim
            if observation.shape != (obs_dim,) or not np.isfinite(observation).all():
                raise ValueError(f"Expected {obs_dim} finite observation features")
            if not s.space.valid(proposal):
                raise ValueError("Proposal violates hard action domain")
            decision = self.controller.decide(observation, proposal, self.slot)
            response = {
                "version": 1,
                "slot": self.slot,
                "action": decision.action.tolist(),
                "fallback": decision.fallback,
                "intervention": decision.intervention,
                "projection_distance": decision.distance,
                "reason": decision.reason,
            }
            self.pending = {
                "slot": self.slot,
                "observation": observation.tolist(),
                "proposal": proposal.tolist(),
                "projected": decision.action.tolist(),
            }
            self.slot += 1
        else:
            raise ValueError("Unknown request type")
        if record is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.last_request, self.last_response = canonical, response
        return response
