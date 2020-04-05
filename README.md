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
- train acc: 0.4490127569675794
- dev acc: 0.458872491568801
- test acc: 0.5678595543551654

#### cnn-one-fpool3
- train acc: 0.6766068091330137
- dev acc: 0.6844459326939176
- test acc: 0.7611973891514743

#### cnn-one-fstride4
- train acc: 0.5782346090300913
- dev acc: 0.5868353703747997
- test acc: 0.6754445194688273

#### cnn-one-fstride8
- train acc: 0.3323853633433547
- dev acc: 0.33911358798344854
- test acc: 0.39297771775827145

#### cnn-tstride2
- train acc: 0.5028845372552206
- dev acc: 0.5161328900475974
- test acc: 0.6378573036236777

#### cnn-tstride4
- train acc: 0.4780206386609247
- dev acc: 0.49130569972972327
- test acc: 0.6054467702003151

#### cnn-tstride8
- train acc: 0.3097424230112944
- dev acc: 0.32103135688488127
- test acc: 0.42493810488408734

#### cnn-tpool2
- train acc: 0.6075675090057149
- dev acc: 0.6197469444378004
- test acc: 0.7254107584965114

#### cnn-tpool3
- train acc: 0.5799680398689093
- dev acc: 0.5925757611997416
- test acc: 0.6975016880486158
