import pandas as pd


def main():

    # TODO: Make Series using pandas Series()
    module_score_dic = {'Database': 90, 'Security': 70, 'Math': 100, 'Machine Learning': 80}
    module_score = pd.Series(module_score_dic)
    print("Module_score: \n", module_score, '\n')
    print("type of module_score: \n", type(module_score))

    # TODO: Make DataFrame using pandas DataFrame()
    # data = pd.DataFrame(module_score, columns=['score'])
    data = pd.DataFrame(module_score, index=[x for x in module_score.keys()], columns=['score'])
    print("data: \n", data, '\n')
    print("type of data: \n", type(data))


if __name__ == "__main__":
    main()
