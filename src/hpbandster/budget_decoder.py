class BudgetDecoderBase:
    def __init__(self, **kwargs):
        pass

    def adjusted_arguments(self, arguments, budget):
        return [arguments]


class SimpleBudgetDecoder(BudgetDecoderBase):
    """
    This class implements a simple budget for HyperBand [Cite Hyperband].
    Budget is decoded in a specified way:
        - budget = 1: Train (20 minutes by default) on 67% of the data, report results on the 33% validation set.
        - budget = 3: Use the same settings as budget 1 but train 3 times longer (60 minutes by default).
        - budget = 9: Train 3 models on 3 CV folds using the same training time limit for each fold as for budget 3.
        - budget = 27: Same as budget 9 but use 9 CV folds instead.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def adjusted_arguments(self, arguments, budget):
        """
        Main function for this class. Based on the budget returns a list of adjusted arguments that are used to
        perform training runs. Number of folds and training time is adjusted.

        Args:
            arguments: ExperimentArguments object containing current run parameters
            budget: Value of the budget as requested by the Architecture Search Optimizer, implemented: {1, 3, 9, 27}
        Returns:
            A list with adjusted ExperimentArguments objects, later used for the training.
        """
        budget = int(budget)
        adjusted_arguments_list = []

        # Basic train with one validation fold (33%)
        if budget == 1:
            adjusted_arguments = arguments.copy()
            adjusted_arguments.cv_n = 3
            adjusted_arguments.cv_k = 2
            adjusted_arguments_list.append(adjusted_arguments)

        # Increase the time x3
        elif budget == 3:
            adjusted_arguments = arguments.copy()
            adjusted_arguments.budget = arguments.budget * 3
            adjusted_arguments_list.append(adjusted_arguments)

        # Make 3 fold CV or 9 fold CV
        elif budget == 9 or budget == 27:
            for c_n, c_k in [(budget//3, i) for i in range(budget//3)]:
                adjusted_arguments = arguments.copy()
                adjusted_arguments.cv_n = c_n
                adjusted_arguments.cv_k = c_k
                adjusted_arguments.budget = arguments.budget * 3
                adjusted_arguments_list.append(adjusted_arguments)

        return adjusted_arguments_list
