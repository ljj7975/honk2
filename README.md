# honk2

## dataset

### GoogleSpeechCommandsDataset
- version1: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
- version2: `wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`

### HeySnipsDataset
https://github.com/snipsco/keyword-spotting-research-datasets

## tensorboard
`tensorboard --logdir=<log dir>`

## Performance Summary

### ResNet

#### res8
- train acc: 0.9447739768695322
- dev acc: 0.9431462125379703
- test acc: 0.9365293720459149

#### res15
- train acc: 0.9701254028872457
- dev acc: 0.9686191968236504
- test acc: 0.9615124915597569

#### res26
- train acc: 0.9627312369654126
- dev acc: 0.9620655839651749
- test acc: 0.9540850776502363

### CNN

#### cnn-trad-pool2
- train acc: 0.8374908588608109
- dev acc: 0.8369011456863354
- test acc: 0.8505514292144947

#### cnn-trad-fpool3
- train acc: 0.9124617426396902
- dev acc: 0.9103810076394933
- test acc: 0.8936170212765957

#### cnn-one-fpool3
- train acc: 0.829446656374421
- dev acc: 0.8299245720916739
- test acc: 0.8279459901800328

#### cnn-one-fstride4
- train acc: 0.7669077218926897
- dev acc: 0.7701624601102408
- test acc: 0.7700490998363339

#### cnn-one-fstride8
- train acc: 0.5886351940629994
- dev acc: 0.5926651194275215
- test acc: 0.6121112929623568

#### cnn-tstride2
- train acc: 0.8917959968581566
- dev acc: 0.8924910550236921
- test acc: 0.8056464811783961

#### cnn-tstride4
- train acc: 0.8786056715690258
- dev acc: 0.8808142346001354
- test acc: 0.8966857610474632

#### cnn-tstride8
- train acc: 0.8433953576555347
- dev acc: 0.8475485929794023
- test acc: 0.877454991816694

#### cnn-tpool2
- train acc: 0.9132472034885296
- dev acc: 0.9126776907455758
- test acc: 0.9065057283142389

#### cnn-tpool3
- train acc: 0.9036320793044609
- dev acc: 0.9034184314863166
- test acc: 0.9095744680851063
