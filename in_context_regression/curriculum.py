import math

class CurriculumFullArgs:
    def __init__(self, 
        training_curriculum_dims_start,
        training_curriculum_dims_end,
        training_curriculum_dims_inc,
        training_curriculum_dims_interval,
        training_curriculum_points_start,
        training_curriculum_points_end,
        training_curriculum_points_inc,
        training_curriculum_points_interval,
    ):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = training_curriculum_dims_start
        self.n_points = training_curriculum_points_start
        self.n_dims_schedule = {
            'start': training_curriculum_dims_start, 
            'end': training_curriculum_dims_end, 
            'inc': training_curriculum_dims_inc, 
            'interval': training_curriculum_dims_interval
        }
        self.n_points_schedule = {
            'start': training_curriculum_points_start, 
            'end': training_curriculum_points_end, 
            'inc': training_curriculum_points_inc, 
            'interval': training_curriculum_points_interval
        }
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule['interval'] == 0:
            var += schedule['inc']

        return min(var, schedule['end'])



class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_points = args.points.start
        self.n_dims_schedule = args.dims
        self.n_points_schedule = args.points
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)
