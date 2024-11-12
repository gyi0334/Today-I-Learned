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

## 2.Related work

### 2.1 Convolutional neural networks

Hubel과 Wiesel이 고양이 시각 피질에 대해 발견한 획기적인 연구[23][22]에 영감을 받아 Fukushima[13]는 단순 유닛 층과 복합 유닛 층이 쌍을 이루어 쌓여있는 구조인 Neocognitron이라는 계층적 모델을 제안했다. 최초의 CNN은 LeCun 등[28][27]에 의해 제안되었습니다. 기본적으로 CNN은 단순 유닛의 수용 영역을 학습하기 위해 역전파(Back-propagation, BP)알고리즘을 통합함으로서 Neocognitron과 차이가 있다. 탄생 이후 CNN은 국소 연결, 가중치 공유 및 국소 pooling의 특징을 가진다. 첫 번째와 두 번째 특징은 모델이 다층 퍼셉트론(MLP)보다 적은 매개변수로 유용한 지역 사각 패턴을 발견할 수 있게 한다. 세 번째 특성은 네트워크에 어느 정도의 변환 불변성(translation invariance)을 부여한다. Saxe 등[41]의 연구에 따르면, CNN의 뛰어난 성능은 이러한 특성들에 크게 기인하며, 무작위 가중치를 가진 특정 구조도 좋은 결과를 낼 수 있다고 한다. 

지난 몇 년 동안 CNN의 성능을 향상시키기 위한 다양한 기술들이 개발되었다. ReLU 함수(Rectified Linear Function) [14]는 역전파(BP) 알고리즘에서 흔히 발생하는 기울기 소실 문제(gradient vanishing effect)에 강한 저항성을 지녀, 가장 일반적으로 사용되는 활성화 함수가 되었다. 드롭아웃(Dropout) [48]은 학습 과정에서 신경망이 과적합되는 것을 방지하는 효과적인 기법이다. Goodfellow 등[17]은 드롭아웃의 모델 평균화 기능을 활용하기 위해 활성화 함수로 특징 채널에 대한 최대 풀링(max pooling)을 사용했다. 합성곱 유닛의 비선형성을 강화하기 위해, Lin 등[33]은 Network in Network (NIN) 구조를 제안했으며, 여기서 합성곱 연산은 입력 특징 맵 위를 슬라이딩하는 국소 다층 퍼셉트론(MLP) [39]으로 대체되었다. NIN이 과적합되는 것을 방지하기 위해 완전 연결 층 대신 전역 평균 풀링 층(global average pooling layer)이 사용되었다. Simonyan과 Zisserman[44]은 작은 필터의 스택이 동일한 매개변수를 가진 큰 필터보다 더 강한 비선형성을 갖는다는 점을 고려하여, 3×3 합성곱을 사용해 매우 깊은 네트워크를 구축했다. Szegedy 등[50]은 다중 스케일 인셉션 모듈(multi-scale inception modules)을 제안하고 이를 바탕으로 GoogLeNet을 구축했으며, 이 모델에서도 작은 필터가 선호되었다. CNN은 계산 집약적인 모델이어서 CPU로 실행하기 어려운 경우가 많다. GPU의 사용은 CNN의 대규모 데이터셋에 대한 학습과 테스트를 크게 용이하게 했다. CNN의 첫 성공적인 GPU 구현은 ImageNet 대규모 시각 인식 챌린지(ILSVRC) 2012에서 우승한 AlexNet [26]이며, 그 이후 매년 이 대회에 출품된 대부분의 모델은 GPU를 기반으로 한 CNN이다.

### 2.2 Recurrent neural networks

순환 신경망(Recurrent neural network, RNN)은 인공 신경망 분야에서 오랜 역사를 가지고 있다.[4, 21, 11, 37, 10, 24] 그러나 RNN의 성공적인 응용 대부분은 필기 인식[18]이나 음성 인식[19]과 같은 순차 데이터(sequentail data) 모델링에 관련되어 있다. 정적 시각 신호 처리에 RNN을 적용한 몇 가지 연구가 아래에서 간략히 소개된다.

[20]에서는 다차원 RNN(Multi-dimensional RNN, MDRNN)을 오프라인 필기 인식을 위해 제안했다. MDRNN은 이미지를 2차원 순차 데이터로 처리하는 방향성 구조를 가지고 있다. 또한, MDRNN은 단일 은닉층을 가지므로 CNN과 같은 특징 계층 구조를 생성할 수 없다.

[2]에서는 Neural Abstraction Pyramid(NAP)이라는 계층적 RNN을 이미지 처리를 위해 제안했다. NAP은 생물학에서 영감을 받은 아키텍처로 수직 및 횡방향 순환 연결을 통해 이미지 해석이 점진적으로 정제되어 시각적 모호성을 해결한다. 구조 설계 시, 생물학적 타당성이 강조되었다. 예를 들어, 대부분의 딥러닝 모델에서는 고려되지 않는 흥분성 유닛과 억제성 유닛(excitatory and inhibitory unit)을 사용한다. 더 중요한 것은, NAP의 일반적인 프레임 워크에는 순환 및 피드백 연결이 있지만 객체 인식을 위해서는 feed-forward version만 테스트되었다. 순환 NAP은 이미지 재구성(image reconstruction)과 같은 다른 작업에 사용되었다.

NAP 외외에도 일부 다른 계층적 모델에선 top-down 연결이 사용되었다. Lee 등[31]은 비지도 특징 학습을 위해 CDBN(Convolutional deep belief network)을 제안했다. 추론 과정에서 최상위 층의 정보가 중간 층을 거쳐 최하위 층으로 전달될 수 있다. 이 층별 전파 아이디어와는 달리, Pinheiro와 Collobert[36]은 CNN의 최상위 층에서 최하위 층으로 직접 연결되는 추가 연결을 사용했다. 이 모델은 장면 레이블링(scene labeling)에 사용되었다. 이러한 모델들은 RCNN과 다르다. RCNN에서는 동일한 층 내에서 순환 연결이 존재하며 층 간 연결이 아니다.

RCNN과 일부 코딩 모델(Sparse coding models) 간에는 흥미로운 관계가 있습니다.[15] 이 모델들에서는 고정점 업데이트(Fixed-point updates)가 추론에 사용되며, 반목 최적화 과정이 암묵적으로 순환 신경망을 정의한다. 지도 학습 기법이 희소 코딩 모델의 비지도 학습 프레임 워크에 통합될 수 있음을 유의하세요[3] 그러나 이러한 기법이 희소 코딩 모델을 객체 인식을 위한 CNN과 경쟁할 수 있을만큼 성능을 높이지 못했다.

마지막으로 우리의 모델은 재귀 층이 동일한 가중치를 공유하는 층의 스택으로 펼쳐지는 재귀 신경망(Recursive neural network)과도 관련이 있다.[46] Socher 등[45]은 장면 구문 분석(Scene parsing)을 수행하기 위해 재귀 신경망을 사용했다. Eigen 등[9]은 재귀 합성곱 신경망을 사용하여 CNN 성능에 영향을 미치는 요인들을 연구했으며, 이는 RCNN의 시간-전개(time-unfolded) 버전과 동일하지만, 각 전개된 층에 feed-forward 입력이 없는 구조이다.

## 3. RCNN Model

### 3.1. Recurrent convolutional layer

RCNN의 핵심 모듈은 recurrent convolutional layer(재귀적 합성곱 계층, RCL)이다. RCL 유닛의 상태는 이산 시간 단계에서 진화한다. RCL에서 𝑘번째 특징 맵에서 위치 
(𝑖,𝑗)에 있는 유닛의 시간 단계 𝑡에서의 순 입력 𝑧𝑖𝑗𝑘(𝑡)는 다음과 같이 주어진다.