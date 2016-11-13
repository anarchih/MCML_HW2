MCML_HW2
==============

```
python3 main.py training/ test/ [model_method] [kernel_type] [feature_scaling]
```

##### model_method
- 1: create 26 two-class classifiers
- 2: use the default model

##### kernel_type
- linear
- poly
- rbf
- sigmoid

##### feature_scaling
- 1: rescaling the range in [0, 1]
- 2: standardization
- 3: normalization (L2 norm)

# Result of two-class classifier
### linear
    rescaling the range in [0, 1]   standardization         normalization (L2 norm)
       precision   recall           precision   recall      precision   recall
    A     1.0000   1.0000              1.0000   1.0000         1.0000   0.9600
    B     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    C     0.8571   0.9600              0.0000   0.0000         0.0000   0.0000
    D     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    E     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    F     0.9615   1.0000              1.0000   1.0000         0.0000   0.0000
    G     1.0000   0.8000              1.0000   0.1800         0.0000   0.0000
    H     1.0000   1.0000              0.0000   0.0000         0.0000   0.0000
    I     0.0000   0.0000              0.0000   0.0000         1.0000   1.0000
    J     0.9800   0.9800              0.0000   0.0000         0.0000   0.0000
    K     0.7885   0.8200              0.9804   1.0000         0.0000   0.0000
    L     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    M     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    N     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    O     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    P     0.9556   0.8600              0.0000   0.0000         1.0000   1.0000
    Q     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    R     0.0000   0.0000              0.8136   0.9600         0.0000   0.0000
    S     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    T     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    U     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    V     0.7077   0.9200              0.0000   0.0000         0.0000   0.0000
    W     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    X     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    Y     0.0000   0.0000              0.0000   0.0000         1.0000   1.0000
    Z     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    --------------------------------------------------------------------------
    avg   0.3942   0.3977              0.2613   0.2362         0.1538   0.1523

### poly
    rescaling the range in [0, 1]   standardization         normalization (L2 norm)
       precision   recall           precision   recall      precision   recall
    A     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    B     0.8929   1.0000              0.1176   0.0800         0.0000   0.0000
    C     0.8772   1.0000              0.7143   0.3000         0.0000   0.0000
    D     1.0000   0.3200              0.7632   0.5800         0.0000   0.0000
    E     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    F     0.7692   1.0000              1.0000   1.0000         0.0000   0.0000
    G     1.0000   0.9200              0.9434   1.0000         0.0000   0.0000
    H     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    I     0.9804   1.0000              0.9091   1.0000         0.0000   0.0000
    J     0.8621   1.0000              0.9231   0.9600         0.0000   0.0000
    K     0.9592   0.9400              0.9804   1.0000         0.0000   0.0000
    L     1.0000   0.9600              0.1071   0.0600         0.0000   0.0000
    M     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    N     0.9000   0.9000              1.0000   1.0000         0.0000   0.0000
    O     1.0000   1.0000              0.9804   1.0000         0.0000   0.0000
    P     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    Q     0.7656   0.9800              0.8491   0.9000         0.0000   0.0000
    R     0.0000   0.0000              1.0000   1.0000         0.0000   0.0000
    S     0.7797   0.9200              0.8936   0.8400         0.0000   0.0000
    T     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    U     0.7778   0.8400              0.8140   0.7000         0.0000   0.0000
    V     0.8621   1.0000              1.0000   1.0000         0.0000   0.0000
    W     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    X     0.9615   1.0000              0.0000   0.0000         0.0000   0.0000
    Y     0.8235   0.8400              0.8400   0.8400         0.0000   0.0000
    Z     0.8621   1.0000              0.8727   0.9600         0.0000   0.0000
    --------------------------------------------------------------------------
    avg   0.8105   0.8315              0.7580   0.7392         0.0000   0.0000

### rbf
    rescaling the range in [0, 1]   standardization         normalization (L2 norm)
       precision   recall           precision   recall      precision   recall
    A     1.0000   1.0000              1.0000   0.9400         1.0000   1.0000
    B     0.9615   1.0000              1.0000   1.0000         0.0000   0.0000
    C     1.0000   1.0000              1.0000   0.9400         0.0000   0.0000
    D     1.0000   1.0000              1.0000   0.9800         0.0000   0.0000
    E     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    F     0.9615   1.0000              1.0000   1.0000         0.0000   0.0000
    G     1.0000   0.9600              1.0000   1.0000         0.0000   0.0000
    H     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    I     1.0000   1.0000              1.0000   1.0000         1.0000   1.0000
    J     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    K     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    L     1.0000   0.9600              1.0000   1.0000         0.0000   0.0000
    M     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    N     1.0000   0.9400              1.0000   0.9600         0.0000   0.0000
    O     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    P     1.0000   1.0000              1.0000   1.0000         1.0000   1.0000
    Q     1.0000   1.0000              0.9434   1.0000         0.0000   0.0000
    R     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    S     0.9615   1.0000              1.0000   1.0000         0.0000   0.0000
    T     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    U     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    V     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    W     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    X     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    Y     1.0000   1.0000              1.0000   1.0000         1.0000   1.0000
    Z     1.0000   1.0000              1.0000   1.0000         0.0000   0.0000
    --------------------------------------------------------------------------
    avg   0.9956   0.9946              0.9978   0.9931         0.1538   0.1538

### sigmoid
    rescaling the range in [0, 1]   standardization         normalization (L2 norm)
       precision   recall           precision   recall      precision   recall
    A     0.0000   0.0000              0.0000   0.0000         1.0000   0.2200
    B     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    C     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    D     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    E     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    F     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    G     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    H     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    I     0.0000   0.0000              0.0000   0.0000         1.0000   1.0000
    J     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    K     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    L     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    M     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    N     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    O     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    P     0.0000   0.0000              0.0000   0.0000         1.0000   1.0000
    Q     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    R     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    S     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    T     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    U     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    V     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    W     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    X     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    Y     0.0000   0.0000              0.0000   0.0000         1.0000   1.0000
    Z     0.0000   0.0000              0.0000   0.0000         0.0000   0.0000
    -------------------------------------------------------------------------- 
    avg   0.0000   0.0000              0.0000   0.0000         0.1538   0.1238

# Result of default model
### linear
    rescaling the range in [0, 1]   standardization          normalization (L2 norm)
       precision    recall          precision    recall      precision    recall
    A     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    B     0.9615    1.0000             1.0000    1.0000         0.5000    0.0800
    C     0.9423    0.9800             1.0000    0.8600         0.9070    0.7800
    D     1.0000    1.0000             1.0000    1.0000         0.8519    0.4600
    E     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    F     0.9615    1.0000             1.0000    1.0000         0.5814    1.0000
    G     1.0000    0.9600             1.0000    1.0000         1.0000    0.2800
    H     1.0000    1.0000             1.0000    1.0000         1.0000    0.8000
    I     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    J     1.0000    1.0000             1.0000    1.0000         1.0000    0.7200
    K     1.0000    1.0000             1.0000    1.0000         0.7812    1.0000
    L     1.0000    0.9600             1.0000    1.0000         0.8696    0.8000
    M     1.0000    1.0000             1.0000    1.0000         0.6724    0.7800
    N     1.0000    0.9600             1.0000    1.0000         0.5789    0.8800
    O     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    P     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    Q     0.9792    0.9400             0.8772    1.0000         0.8929    1.0000
    R     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    S     0.9615    1.0000             1.0000    1.0000         0.8333    1.0000
    T     1.0000    1.0000             1.0000    1.0000         1.0000    0.9200
    U     1.0000    1.0000             1.0000    1.0000         0.6364    0.7000
    V     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    W     1.0000    1.0000             1.0000    1.0000         0.5529    0.9400
    X     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    Y     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    Z     1.0000    1.0000             1.0000    1.0000         0.5000    1.0000
    ----------------------------------------------------------------------------
    avg   0.9925    0.9923             0.9953    0.9946         0.8138    0.8131

### poly
    rescaling the range in [0, 1]   standardization          normalization (L2 norm)
       precision    recall          precision    recall      precision    recall
    A     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    B     0.9615    1.0000             1.0000    1.0000         0.0000    0.0000
    C     1.0000    1.0000             0.9216    0.9400         0.0000    0.0000
    D     0.9804    1.0000             1.0000    1.0000         0.0000    0.0000
    E     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    F     0.9091    1.0000             1.0000    1.0000         0.0000    0.0000
    G     1.0000    0.9000             1.0000    1.0000         0.0000    0.0000
    H     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    I     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    J     0.9804    1.0000             1.0000    0.9800         0.0000    0.0000
    K     1.0000    0.9800             0.9804    1.0000         0.0000    0.0000
    L     1.0000    0.9600             1.0000    1.0000         0.0000    0.0000
    M     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    N     1.0000    0.9400             1.0000    1.0000         0.0000    0.0000
    O     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    P     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    Q     1.0000    1.0000             0.9412    0.9600         0.0000    0.0000
    R     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    S     0.9615    1.0000             1.0000    0.9600         0.0000    0.0000
    T     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    U     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    V     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    W     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    X     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    Y     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    Z     1.0000    1.0000             1.0000    1.0000         0.0385    1.0000
    ----------------------------------------------------------------------------
    avg   0.9920    0.9915             0.9940    0.9938         0.0015    0.0385

### rbf
    rescaling the range in [0, 1]   standardization          normalization (L2 norm)
       precision    recall          precision    recall      precision    recall
    A     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    B     0.9615    1.0000             1.0000    1.0000         0.0000    0.0000
    C     1.0000    1.0000             1.0000    0.9400         0.9268    0.7600
    D     0.9804    1.0000             1.0000    1.0000         0.7059    0.4800
    E     1.0000    1.0000             1.0000    1.0000         0.0000    0.0000
    F     0.9615    1.0000             1.0000    1.0000         0.5000    1.0000
    G     1.0000    0.9600             1.0000    1.0000         0.0000    0.0000
    H     1.0000    1.0000             1.0000    1.0000         0.9318    0.8200
    I     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    J     1.0000    1.0000             1.0000    1.0000         1.0000    0.7200
    K     1.0000    1.0000             1.0000    1.0000         0.7812    1.0000
    L     1.0000    0.9600             1.0000    1.0000         0.6780    0.8000
    M     1.0000    1.0000             1.0000    1.0000         0.6111    0.6600
    N     1.0000    0.9400             1.0000    1.0000         0.6000    0.7200
    O     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    P     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    Q     1.0000    1.0000             0.9434    1.0000         0.8929    1.0000
    R     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    S     0.9615    1.0000             1.0000    1.0000         0.7692    1.0000
    T     1.0000    1.0000             1.0000    1.0000         1.0000    0.6400
    U     1.0000    1.0000             1.0000    1.0000         0.5147    0.7000
    V     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    W     1.0000    1.0000             1.0000    1.0000         0.5402    0.9400
    X     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    Y     1.0000    1.0000             1.0000    1.0000         1.0000    1.0000
    Z     1.0000    1.0000             1.0000    1.0000         0.5000    1.0000
    ----------------------------------------------------------------------------
    avg   0.9948    0.9946             0.9978    0.9977         0.7289    0.7785

### sigmoid
    rescaling the range in [0, 1]   standardization          normalization (L2 norm)
       precision    recall          precision    recall      precision    recall
    A     0.0000    0.0000             0.0000    0.0000         1.0000    1.0000
    B     0.0678    0.1600             0.0000    0.0000         1.0000    0.0600
    C     0.0000    0.0000             0.0000    0.0000         0.9412    0.6400
    D     0.0000    0.0000             0.0000    0.0000         0.5000    0.5000
    E     0.0000    0.0000             0.0000    0.0000         0.0000    0.0000
    F     0.0000    0.0000             0.0000    0.0000         0.5000    1.0000
    G     0.0000    0.0000             0.0000    0.0000         0.0000    0.0000
    H     0.0000    0.0000             0.0000    0.0000         0.9545    0.8400
    I     0.0000    0.0000             0.0000    0.0000         1.0000    1.0000
    J     0.0000    0.0000             0.0000    0.0000         1.0000    0.7200
    K     0.3182    0.1400             0.0000    0.0000         0.7812    1.0000
    L     0.0000    0.0000             0.0000    0.0000         0.6364    0.8400
    M     0.0000    0.0000             0.0000    0.0000         0.4054    0.6000
    N     0.0000    0.0000             0.0000    0.0000         0.4400    0.2200
    O     0.0000    0.0000             0.0000    0.0000         1.0000    1.0000
    P     0.0000    0.0000             0.0000    0.0000         1.0000    1.0000
    Q     0.2212    0.9200             0.0000    0.0000         0.8475    1.0000
    R     0.0000    0.0000             0.0000    0.0000         1.0000    1.0000
    S     0.0000    0.0000             0.0000    0.0000         0.5435    1.0000
    T     0.0000    0.0000             0.0000    0.0000         1.0000    0.2600
    U     0.0000    0.0000             0.0000    0.0000         0.5385    0.5600
    V     0.0808    0.4200             0.1916    1.0000         0.9091    1.0000
    W     0.0000    0.0000             0.0000    0.0000         0.4578    0.7600
    X     0.0000    0.0000             0.1109    1.0000         1.0000    1.0000
    Y     0.0741    0.1600             0.0000    0.0000         1.0000    1.0000
    Z     0.1961    1.0000             0.1202    0.8200         0.5000    1.0000
    ----------------------------------------------------------------------------
    avg   0.0368    0.1077             0.0163    0.1085         0.7290    0.7308