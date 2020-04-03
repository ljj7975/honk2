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

