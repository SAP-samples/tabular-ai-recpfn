import time


class StageTimer:
    def __init__(self):
        self.stage_times = []

    def start(self):
        self.start_time = time.time()

    def end(self):
        end_time = time.time()
        elapsed = end_time - self.start_time
        self.stage_times.append(elapsed)
        return elapsed

    def get_total_time(self):
        return sum(self.stage_times)


class LogTime:
    def __init__(self):
        self.timers:dict[str, StageTimer] = {}

    def start_stage(self, stage_name: str):
        if stage_name not in self.timers:
            self.timers[stage_name] = StageTimer()
        self.timers[stage_name].start()

    def end_stage(self, stage_name: str):
        if stage_name in self.timers:
            elapsed = self.timers[stage_name].end()
        else:
            raise ValueError(f"No timer found for stage '{stage_name}'.")

    def log_summary(self):
        log_str = "Timing Summary:\n"
        times = {}
        for stage_name, timer in self.timers.items():
            total_time = timer.get_total_time()
            times[stage_name] = total_time
        log_str += "Total time taken for all stages: {:.4f} seconds.\n".format(sum(times.values()))
        for stage_name, total_time in times.items():
            percentage = (total_time / sum(times.values())) * 100 if sum(times.values()) > 0 else 0
            log_str += "Stage '{}': {:.4f} seconds ({:.2f}%)\n".format(stage_name, total_time, percentage)
        return log_str
