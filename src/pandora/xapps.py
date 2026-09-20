"""Three reference black-box policies. Pandora only consumes their joint proposals."""

import numpy as np


class ReferenceXApps:
    def __init__(self, space, simulation):
        self.space, self.config = space, simulation
        self.previous = space.neutral()

    def propose(self, telemetry):
        s, c, t = self.space, self.config, telemetry
        old_rho, old_nu, old_omega = s.split(self.previous)
        need = t.demand + t.queue / c.slot_seconds
        resource_target = s.cells * need / max(need.sum(), 1e-12)
        rho = 0.85 * old_rho + 0.15 * resource_target
        # Recover SINR in dB from B*log2(1+SINR); 3 dB hysteresis.
        sinr = 10 * np.log10(np.maximum(np.exp2(t.capacity / c.bandwidth_mhz) - 1, 1e-12))
        current = old_nu.argmax(axis=1)
        preferred = (sinr - 3 * t.cell_load[None]).argmax(axis=1)
        switch = sinr[np.arange(s.users), preferred] > sinr[np.arange(s.users), current] + 3
        target_cell = np.where(switch, preferred, current)
        target_nu = np.eye(s.cells)[target_cell]
        nu = 0.9 * old_nu + 0.1 * target_nu
        rate = np.array(c.min_rate_mbps)[t.slices]
        delay_limit = np.array(c.max_delay_ms)[t.slices]
        urgency = np.clip(
            0.5
            + 0.25 * (rate - t.throughput) / rate
            + 0.25 * (t.delay - delay_limit) / delay_limit,
            0,
            1,
        )
        omega = 0.8 * old_omega + 0.2 * urgency
        raw = s.normalize(s.join(rho, nu, omega))
        # Each black box remembers its own proposal; it is not trained by the coordinator.
        self.previous = raw
        neutral = s.neutral()
        return s.normalize(neutral + c.aggressiveness * (raw - neutral))
