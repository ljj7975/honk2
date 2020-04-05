# Summary of Models

input tensor size = (32, 40)
kernel = (m,r)
number of channels (features) = n
pool = (p,q)
stride = (s,v)

## Base Model

### cnn-trad-fpool3

|  type |  m | r |  n  | s | v | p | q | params |
|:-----:|:--:|:-:|:---:|:-:|:-:|:-:|:-:|:------:|
|  conv | 20 | 8 |  64 | 1 | 1 | 1 | 3 |  10.2K |
|  conv | 10 | 4 |  64 | 1 | 1 | 1 | 1 | 164.8K |
|  lin  |  - | - |  32 | - | - | - | - |  65.5K |
|  dnn  |  - | - | 128 | - | - | - | - |  4.1K  |
|  lin  |  - | - |  12 | - | - | - | - |  1.5K  |
| total |  - | - |  -  | - | - | - | - | 245.2K |

## Limiting Multiplies

### cnn-one-fpool3

|  type |  m | r |  n  | s | v | p | q | params |
|:-----:|:--:|:-:|:---:|:-:|:-:|:-:|:-:|:------:|
|  conv | 32 | 8 |  54 | 1 | 1 | 1 | 3 |  13.8K |
|  lin  |  - | - |  32 | - | - | - | - |  19.8K |
|  dnn  |  - | - | 128 | - | - | - | - |  4.1K  |
|  dnn  |  - | - | 128 | - | - | - | - |  16.4K |
|  lin  |  - | - |  12 | - | - | - | - |  1.5K  |
| total |  - | - |  -  | - | - | - | - |  54.8K |

### cnn-one-fstride4

|  type |  m | r |  n  | s | v | p | q | params |
|:-----:|:--:|:-:|:---:|:-:|:-:|:-:|:-:|:------:|
|  conv | 32 | 8 | 186 | 1 | 4 | 1 | 3 |  47.6K |
|  lin  |  - | - |  32 | - | - | - | - |  17.9K |
|  dnn  |  - | - | 128 | - | - | - | - |  4.1K  |
|  dnn  |  - | - | 128 | - | - | - | - |  16.4K |
|  lin  |  - | - |  12 | - | - | - | - |  1.5K  |
| total |  - | - |  -  | - | - | - | - |  87.5K |

### cnn-one-fstride8

|  type |  m | r |  n  | s | v | p | q | params |
|:-----:|:--:|:-:|:---:|:-:|:-:|:-:|:-:|:------:|
|  conv | 32 | 8 | 336 | 1 | 8 | 1 | 3 |  86.6K |
|  lin  |  - | - |  32 | - | - | - | - |  10.2K |
|  dnn  |  - | - | 128 | - | - | - | - |  4.1K  |
|  dnn  |  - | - | 128 | - | - | - | - |  16.4K |
|  lin  |  - | - |  12 | - | - | - | - |  1.5K  |
| total |  - | - |  -  | - | - | - | - | 118.8K |


## Limiting Parameters - Sliding in Time

### cnn-tstride2

|  type |  m | r |  n  | s | v | p | q | params |
|:-----:|:--:|:-:|:---:|:-:|:-:|:-:|:-:|:------:|
|  conv | 16 | 8 |  78 | 2 | 1 | 1 | 3 |  10.0K |
|  conv |  9 | 4 |  78 | 1 | 1 | 1 | 1 | 219.0K |
|  lin  |  - | - |  32 | - | - | - | - |  19.8K |
|  dnn  |  - | - | 128 | - | - | - | - |  4.1K  |
|  dnn  |  - | - | 128 | - | - | - | - |  16.4K |
|  lin  |  - | - |  12 | - | - | - | - |  1.5K  |
| total |  - | - |  -  | - | - | - | - | 325.6K |

### cnn-tstride4

|  type |  m | r |  n  | s | v | p | q | params |
|:-----:|:--:|:-:|:---:|:-:|:-:|:-:|:-:|:------:|
|  conv | 16 | 8 | 100 | 4 | 1 | 1 | 3 |  12.8K |
|  conv |  5 | 4 |  78 | 1 | 1 | 1 | 1 | 200.0K |
|  lin  |  - | - |  32 | - | - | - | - |  25.6K |
|  dnn  |  - | - | 128 | - | - | - | - |  4.1K  |
|  dnn  |  - | - | 128 | - | - | - | - |  16.4K |
|  lin  |  - | - |  12 | - | - | - | - |  1.5K  |
| total |  - | - |  -  | - | - | - | - | 260.4K |

### cnn-tstride8

|  type |  m | r |  n  | s | v | p | q | params |
|:-----:|:--:|:-:|:---:|:-:|:-:|:-:|:-:|:------:|
|  conv | 16 | 8 | 126 | 8 | 1 | 1 | 3 |  16.1K |
|  conv |  5 | 4 |  78 | 1 | 1 | 1 | 1 | 190.5K |
|  lin  |  - | - |  32 | - | - | - | - |  32.2K |
|  dnn  |  - | - | 128 | - | - | - | - |  4.1K  |
|  dnn  |  - | - | 128 | - | - | - | - |  16.4K |
|  lin  |  - | - |  12 | - | - | - | - |  1.5K  |
| total |  - | - |  -  | - | - | - | - |  26.8K |

## Limiting Parameters - Pooling in Time

### cnn-tpool2

|  type |  m | r |  n  | s | v | p | q | params |
|:-----:|:--:|:-:|:---:|:-:|:-:|:-:|:-:|:------:|
|  conv | 21 | 8 |  94 | 1 | 1 | 2 | 3 |  5.6M  |
|  conv |  6 | 4 |  94 | 1 | 1 | 1 | 1 |  1.8M  |
|  lin  |  - | - |  32 | - | - | - | - |  65.5K |
|  dnn  |  - | - | 128 | - | - | - | - |  4.1K  |
|  dnn  |  - | - | 128 | - | - | - | - |  16.4K |
|  lin  |  - | - |  12 | - | - | - | - |  1.5K  |
| total |  - | - |  -  | - | - | - | - |   7.5M |

### cnn-tpool3

|  type |  m | r |  n  | s | v | p | q | params |
|:-----:|:--:|:-:|:---:|:-:|:-:|:-:|:-:|:------:|
|  conv | 15 | 8 |  94 | 1 | 1 | 3 | 3 |   7.1M |
|  conv |  6 | 4 |  94 | 1 | 1 | 1 | 1 |   1.6M |
|  lin  |  - | - |  32 | - | - | - | - |  65.5K |
|  dnn  |  - | - | 128 | - | - | - | - |  4.1K  |
|  dnn  |  - | - | 128 | - | - | - | - |  16.4K |
|  lin  |  - | - |  12 | - | - | - | - |  1.5K  |
| total |  - | - |  -  | - | - | - | - |   8.8M |