# Recurrent Convolutional Neural Network for Object Recognition

### Abstract

최근 몇년간 CNN(Convolutional Neural Network)은 컴퓨터 비전 분야에서 굉장한 성공을 거뒀다. 뇌 과학에서 영감을 얻어 CNN은 뇌의 시작부분과 많은 속성을 공유한다. CNN과의 중요한 차이는 CNN은 순전파 구조이지만 Visual system에선 반복적인 연결이 풍부하다는 것이다. 이 사실에서 영감을 얻어, 우리는 각각의 Convolution Layer를 통해 통합되는 연결에 의한 Object recognition을 위한 recurrent CNN(RCNN)을 제안한다. 고정된 입력을 통해 이웃된 활성화한 유닛과 연동된 활성된 RCNN유닛은 시간에 따라 진화한다. 이 이론은 객체 인식에 중요한 정보를 통합하는 모델의 능력을 향상 시켰다. 다른 recurrent neural network들 처럼, 시간에 따른 RCNN의 전개는 고정된 수의 parameter를 가진 임의의 깊은 네트워크를 생성 할 수 있었다. 그 뿐만 아니라, 펼쳐진 네트워크는 학습을 용이하게 하는 여러개의 경로를 가지고 있다. 모델은 네개의 유명한 object recognition dataset 위에서 테스트 되었다. : CIFAR-10, CIFAR-100, MNIST and SVHN. 학습 가능한 parameter가 적은 RCNN은 다른 모델들 보다 모든 dataset에서 성능이 뛰어나다. parameter의 증가는 더욱 좋은 performance를 보여준다. 이러한 결과는 object recognition에서 recurrent structure의 장점을 증명한다.

## 1.Introduction

최근 몇 년 동안 컴퓨터 비전 분야에서 합성곱 신경망(Convolutional Neural Network, CNN)이 급격히 발전해왔습니다. CNN은 여러 벤치마크 데이터셋에서 객체 인식 정확도를 획기적으로 높였으며, 예를 들어 ImageNet 데이터셋의 120만 개 이미지로 학습된 CNN 모델은 1000개의 객체 범주를 기존의 수작업 특징 기반 모델보다 훨씬 높은 성능으로 분류해 냈습니다. 또한, 사전 학습된 CNN 모델의 특징들은 다른 데이터셋에서도 우수한 결과를 보였습니다​

CNN은 1943년 제안된 첫 인공 뉴런에서 기원한 인공 신경망의 한 유형으로, 신경과학에서 유래되었다. CNN 및 기타 계층 모델(Neocognitron 및 HMAX)은 1차 시각 피질(v1)에서 단순 세포와 복합 세포에 대한 Hubel과 Wiesel의 연구와 밀접한 관련이 있다. 이 모델들은 모두 순방향구조(feed-forward architecture)를 가지며, 이는 생물학적 신경망을 단순화한 형태로 볼 수 있다. 그러나 해부갛적 증거에 따르면 대뇌 피질에는 순환 연결(Recurrent Connection)이 널리 존재하며, 이러한 순환 시냅스가 Feed-forward 및 Feedback 시냅스보다 더 많다. 순환 시냅수와 피드백 시냅스 덕분에 객체 인식은 입력이 정적임에도 불구하고 사실 동적 과정으로 이루어진다.

시각 신호의 처리는 주변 맥락에 의해 강하게 조절되며, 이러한 맥락 조절은 개별 뉴런의 반응에서도 관찰된다.[12] 예를 들어, V1 뉴런의 반응 특성은 수용 영역(receptive field, RF) 주변의 맥락에 따라 다양하게 변경될 수 있다.[42]

객체 인식에서 맥락 정보는 중요하다.(Figure 1) feed-forward model은 상위 층에서만 맥락 정보(the face in Figure 1)를 포착할 수 있지만, 이 정보는 더 작은 객체를 인식하는 하위 층의 유닛 활동을 조절하지는 못한다.(the nose in Figure 1) 이를 해결하기 위해, 상향 연결(top-down connections or feedback)을 사용해 상위 층의 정보를 하위 층으로 전파하는 방법(합성곱 심층 신뢰망, Convolutional deep belief network, CDBN)이 있지만 본 연구에서는 다른 접근법으로 동일한 층 내에서의 연결을 사용하는 전략을 제안한다. 

본 논문에서는 정적 객체 인식을 위한 순환 합성곱 신경망(recurrent CNN)을 제안한다. 구조는 Figure 2에 있으며, Feed-forward와 순환 연결이 모두 국소적인 연결을 가지며 서로 다른 위치에서 weight를 공유한다. 이 구조는 동적 제어에 자주 사용되는 순환 다층 퍼셉트론(Recurrent Multilayer Perceptron, RMLP)과 매우 비슷하다.(Figure 2, middle) 주요 차이점은 RMLP의 전체 연결이 MLP와 CNN간의 차이와 마찬가지로 공유된 국소 연결로 대체된 것이다.[40] 이러한 이유료, 제안된 모델을 순환 합성곱 신경망(RCNN)으로 명하였다.

제안돤 RCNN은 여러 object recognition dataset에서 테스트 되었다. RCNN은 더 적은 parameter를 사용하면서 모든 데이터 셋에서 기존의 최신 CNN보다 더 좋은 결과를 달성했으며, 이는 CNN에 비해 RCNN의 우수성을 입증한다. 나머지 내용은 다음과 같이 구성된다. 2장에서는 관련 연구를 검토하고, 3장에서는 RCNN의 아키텍처를 설명한다. 4장에서는 실험 결과와 분석을 제시하며, 마지막으로 5장에서 본 논문을 결론짓는다.