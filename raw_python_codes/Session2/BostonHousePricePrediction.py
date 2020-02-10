"""
회귀는 여러 요인들과 특정 종속 변수 간의 관계를 예측하는 것으로써, 이를 통해 주가 예측 등 여러 예측 과제를 수행할 수 있습니다.

이번 시간에는 사이킷런 라이브러리에 내장되어 있는 데이터셋 중 하나인 Boston Data(보스턴 주택가격 예측을 위한 데이터)를 이용해 보겠습니다. 여러 변수들을 바탕으로 주택 가격(PRICE)을 예측하는 회귀 모델을 만들고 평가하여 봅시다.

유용한 함수들

sklearn.metrics.mean_squared_error: 두 값의 차의 제곱을 구합니다.

np.sqrt(x): x의 제곱근을 구합니다.

linear_model.coef_ : 선형모델의 상관계수를 출력합니다.

코드 구성

이번 예제의 코드는 함수의 형태로 구성되어 있습니다. 함수형 코드 구성은 입력을 받아 결과를 출력하는 형태의 함수들로 코드를 구성하고 전개하는 것을 말합니다.

코드가 길어지거나 반복되는 경우, 함수의 형태로 코드를 구성하는 것이 효율성이나 가독성 면에서 더 적합하다고 할 수 있습니다.

load_data(): 데이터를 불러옵니다.
load_data_target(dataset): 데이터셋을 data와 target을 합하여 DataFrame을 만들고 반환합니다.
plotting_graph(DataFrame): 데이터를 시각화한다.
data2train_eval(DataFrame): 데이터를 Test와 Validation으로 나누어주는 함수를 만들어 봅니다.
main(): 위의 함수들을 이용하여 선형회귀를 진행합니다. (전체 진행되는 과정을 main함수 안에서 확인할 수 있습니다.)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_data():
    return datasets.load_boston()


##데이터셋에 data과 target 합한 DataFrame을 반환하는 함수를 만들어 봅니다.
def load_data_target(dataset):
    ##데이터셋의 Data를 DataFrame으로 만들어 줍니다.
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    # 1.만들어준 DataFrame에 'PRICE'변수를 추가하고, target 데이터를 넣어준다.
    df['PRICE'] = dataset.target
    return df


##데이터를 Seaborn을 이용하여 시각화한다.
def plotting_graph(df):
    # 2.Matplotlib.pyplot의 Subplots를 이용하여 2행 4열의[figsize=(16,8)] 그래프를 생성합니다.
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    ##특징들의 이름을 설정합니다.
    features = ['RM', 'ZN', 'INDUS', 'NOX', 'AGE', 'PTRATIO', 'LSTAT', 'RAD']

    ##Seaborn을 이용하여 특징들과 그 인덱스를 바탕으로 데이터를 시각화 합니다.
    for i, feature in enumerate(features):
        row = int(i / 4)
        col = i % 4
        ##seaborn.replot을 이용하여 변수와 그에 따른 Regression 그래프를 그려줍니다.
        sns.regplot(x=feature, y='PRICE', data=df, ax=axs[row][col])

    fig.savefig("./Image_Output/BostonPrediction.png")


##데이터를 train/validation 데이터로 나누어주는 함수를 만들어 봅니다.
def data2train_eval(df):
    label_data = df['PRICE']
    input_data = df.drop(['PRICE'], axis=1, inplace=False)

    input_train, input_eval, label_train, label_eval = train_test_split(input_data, label_data, test_size=0.3,
                                                                        random_state=432)

    return input_train, input_eval, label_train, label_eval


##정의된 함수를 바탕으로 데이터를 분리하고 회귀(Regression)를 진행합니다.
def main():
    ## 데이터를 불러옵니다.
    df = load_data()
    ## 불러온 데이터를 DataFrame으로 추가시켜 줍니다.
    df_data_target = load_data_target(df)
    ## (Seaborn)그래프를 띄워서 변수 별로 대략적인 Regression이 어떻게 되는지 확인하여 봅니다.
    plotting_graph(df_data_target)

    input_train, input_eval, label_train, label_eval = data2train_eval(df_data_target)

    ## 회귀모델을 생성합니다.
    linear_model = LinearRegression()
    # 3.학습 데이터를 바탕으로 학습을 진행합니다.
    linear_model.fit(input_train, label_train)
    # 4.검증 데이터를 바탕으로 값을 예측합니다.
    pred = linear_model.predict(input_eval)
    # 5.검증 데이터의 제곱 평균 오차(mean squared error)를 계산합니다.
    mse = mean_squared_error(label_eval, pred)
    ## 검증 데이터의 루트 제곱 평균 오차(rooted mean squared error)를 계산합니다.
    rmse = np.sqrt(mse)

    print('MSE: {0:.3f}, RMSE: {1:.3F}'.format(mse, rmse))
    # LinearRegression 모델의 상관계수를 구하고 출력합니다.
    print('회귀 계수값:', np.round(linear_model.coef_, 1))
    # 구한 상관계수를 큰 순서대로 출력합니다.
    coeff = pd.Series(data=np.round(linear_model.coef_, 1), index=input_train.columns)
    print(coeff.sort_values(ascending=False))


main()