// copied from light-tensor
class LearningRateSchedule
{
public:
	LearningRateSchedule() {}
	~LearningRateSchedule() {}

	virtual double GetLearningRate(uint64_t step, uint32_t epoch) = 0;

private:
	double base_lr_;
	double min_lr_;
	double max_lr_;
	uint64_t warmup_steps_;
	uint32_t warmup_epochs_;
	uint32_t total_epochs_;
	uint64_t steps_per_epoch_;
};

class LinearLearningRateSchedule : public LearningRateSchedule
{
public:
	LinearLearningRateSchedule(double min_lr, double max_lr, uint64_t steps_per_epoch, uint32_t total_epochs)
	{
		min_lr_ = min_lr;
		max_lr_ = max_lr;
		total_epochs_ = total_epochs;
		steps_per_epoch_ = steps_per_epoch;
	}

	double GetLearningRate(uint64_t step, uint32_t epoch = 0)
	{
		double scaler;
		double lr;

		scaler = step / (double)(steps_per_epoch_ * total_epochs_ - 1);

		lr = max_lr_ - (max_lr_ - min_lr_) * scaler;

		if (lr < min_lr_)
		{
			lr = min_lr_;
		}

		return lr;
	}

private:
	double min_lr_;
	double max_lr_;
	uint64_t steps_per_epoch_;
	uint32_t total_epochs_;

};

