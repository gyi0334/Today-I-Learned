# Recurrent Convolutional Neural Network for Object Recognition

### Abstract

  최근 몇년간 CNN(Convolutional Neural Network)은 컴퓨터 비전 분야에서 굉장한 성공을 거뒀다. 뇌 과학에서 영감을 얻어 CNN은 뇌의 시작부분과 많은 속성을 공유한다. CNN과의 중요한 차이는 CNN은 순전파 구조이지만 Visual system에선 반복적인 연결이 풍부하다는 것이다. 이 사실에서 영감을 얻어, 우리는 각각의 Convolution Layer를 통해 통합되는 연결에 의한 Object recognition을 위한 recurrent CNN(RCNN)을 제안한다. 고정된 입력을 통해 이웃된 활성화한 유닛과 연동된 활성된 RCNN유닛은 시간에 따라 진화한다. 이 이론은 객체 인식에 중요한 정보를 통합하는 모델의 능력을 향상 시켰다. 다른 recurrent neural network들 처럼, 시간에 따른 RCNN의 전개는 고정된 수의 parameter를 가진 임의의 깊은 네트워크를 생성 할 수 있었다. 그 뿐만 아니라, 펼쳐진 네트워크는 학습을 용이하게 하는 여러개의 경로를 가지고 있다. 모델은 네개의 유명한 object recognition dataset 위에서 테스트 되었다. : CIFAR-10, CIFAR-100, MNIST and SVHN. 학습 가능한 parameter가 적은 RCNN은 다른 모델들 보다 모든 dataset에서 성능이 뛰어나다. parameter의 증가는 더욱 좋은 performance를 보여준다. 이러한 결과는 object recognition에서 recurrent structure의 장점을 증명한다.

## 1.Introduction