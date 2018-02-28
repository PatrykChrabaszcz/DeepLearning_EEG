class BudgetDecoderBase:
    def __init__(self, **kwargs):
        pass

    def adjusted_arguments(self, arguments, budget):
        return [arguments]


class SimpleBudgetDecoder(BudgetDecoderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def adjusted_arguments(self, arguments, budget):
        budget = int(budget)
        adjusted_arguments_list = []
        if budget == 1:
            adjusted_arguments = arguments.copy()
            adjusted_arguments.cv_n = 3
            adjusted_arguments.cv_k = 2
            adjusted_arguments_list.append(adjusted_arguments)

        elif budget == 3 or budget == 9:
            for c_n, c_k in [(budget, i) for i in range(budget)]:
                adjusted_arguments = arguments.copy()
                adjusted_arguments.cv_n = c_n
                adjusted_arguments.cv_k = c_k
                adjusted_arguments_list.append(adjusted_arguments)
        elif budget == 27:
            for c_n, c_k in [(9, i) for i in range(9)]:
                adjusted_arguments = arguments.copy()
                adjusted_arguments.cv_n = c_n
                adjusted_arguments.cv_k = c_k
                adjusted_arguments.budget = arguments.budget * 3
                adjusted_arguments_list.append(adjusted_arguments)

        return adjusted_arguments_list
