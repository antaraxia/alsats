# alsats - active learning (for a few) satoshis.
<i>Label data intelligently using Active Learning. Pay in sats only for the compute you consume.</i>

alsats reduces the time and cost required to create minimum viable datasets in supervised learning problems.
It has the following features:
* API based model training and data labeling.
* Intelligent labeling - Label training examples based on model uncertainty predictions for said examples.
* Iterative Learning - Learn models starting with as little as one data point. As more data is labeled and trained, model metrics and labeling suggestions improve.
* Flexible payments - Pay ONLY for the compute consumed in the process (i.e. pay for as little as one milisat per training/label iteration).
* Train and label trustlessly - No need to register/sign-up for an account. 
* Data security - alsats doesn't store any user data in it's current implementation. Future implementations will store data only if the customer wants to.

Below are some screenshots from a working demo of the code implemented in Streamlit (demo.py in the repo).
![demo_About](https://user-images.githubusercontent.com/105051775/178170644-3eddcfdb-dda4-47c3-ad11-2e00f641d03d.JPG)
![demo_dash](https://user-images.githubusercontent.com/105051775/178170648-c801e8dd-1111-43c0-9947-01f2f9eeaea9.JPG)
The infrastructure supporting the implementation of the demo is a Lightning Network simnet implemented in <a href="https://lightningpolar.com/">Polar</a>. A snapshot of a local implementation of the lightning simnet is shown below. The API server is hosted locally by alice, who runs her own LND node. bob, a client interested in labeling MNIST images creates an outgoing Lightning channel with alice and begins the labeling process. Upon completion, bob has the option to close the channel and finalize the transaction, whereupon alice's Lightning wallet will contain the sats bob paid for.
![simnet](https://user-images.githubusercontent.com/105051775/178170349-25832556-5db8-4ab9-b395-82edbd591f16.JPG)
A demo video of alsats in action is <a href = "https://www.youtube.com/watch?v=hyjvazfp0uc">now available for viewing</a>. 
