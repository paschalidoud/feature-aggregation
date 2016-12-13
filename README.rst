Local feature aggregation
=========================

This is a library that implements methods to aggregate local features
(mainly for multimedia) into a single global feature that can be used
easily with any classifier.

Dependencies
------------

The library depends on **scikit-learn** and all the feature aggregation
methods extend the scikit-learn BaseEstimator class.

Example
-------

.. code:: python

    import numpy as np
    from feature_aggregation import BagOfWords, FisherVectors

    X = np.random.rand(1000, 2)
    bow = BagOfWords(10)
    fv = FisherVectors(10)

    bow.fit(X)
    fv.fit(X)

    G1 = bow.transform(np.random.rand(10, 100, 2))
    G2 = fv.transform([
        np.random.rand(int(np.random.rand()*100), 2) for _ in range(10)
    ])

A more complex example using OpenCV to extract dense SIFT and then
transform them using Bag Of Words and train an SVM with chi square
additive kernel.

.. code:: python

    import numpy as np
    import cv2
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.kernel_approximation import AdditiveChi2Sampler
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    from feature_aggregation import BagOfWords

    def sift(*args, **kwargs):
        try:
            return cv2.xfeatures2d.SIFT_create(*args, **kwargs)
        except:
            return cv2.SIFT()

    def dsift(img, step=5):
        keypoints = [
            cv2.KeyPoint(x, y, step)
            for y in range(0, img.shape[0], step)
            for x in range(0, img.shape[1], step)
        ]
        features = sift().compute(img, keypoints)[1]
        features /= features.sum(axis=1).reshape(-1, 1)
        return features

    # Generate dense SIFT features
    faces = fetch_olivetti_faces()
    features = [
        dsift((x.reshape(64, 64, 1)*255).astype(np.uint8))
        for x in faces.data
    ]

    # Aggregate those features with bag of words using online training
    bow = BagOfWords(100)
    for i in range(2):
        for j in range(0, len(features), 10):
            bow.partial_fit(features[j:j+10])
    faces_bow = bow.transform(features)

    # Split in training and test set
    train = np.arange(len(features))
    np.random.shuffle(train)
    test = train[200:]
    train = train[:200]

    # Train and evaluate
    svm = Pipeline([("chi2", AdditiveChi2Sampler()), ("svm", LinearSVC(C=10))])
    svm.fit(faces_bow[train], faces.target[train])
    print(classification_report(faces.target[test], svm.predict(faces_bow[test])))
