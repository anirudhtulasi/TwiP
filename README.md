![Total Lines of Code](https://img.shields.io/tokei/lines/github/anirudhtulasi/TwiP?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/anirudhtulasi/TwiP?style=for-the-badge)
![GitHub pull requests](https://img.shields.io/github/issues-pr/anirudhtulasi/TwiP?style=for-the-badge)
![GitHub followers](https://img.shields.io/github/followers/anirudhtulasi?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/anirudhtulasi/TwiP?style=for-the-badge)
![Cocoapods platforms](https://img.shields.io/cocoapods/p/AFNetworking?style=for-the-badge)
![Website](https://img.shields.io/website?style=for-the-badge&url=https%3A%2F%2Fanirudhxtwitter.herokuapp.com%2F)
![GitHub last commit](https://img.shields.io/github/last-commit/anirudhtulasi/TwiP?style=for-the-badge)
![Maintenance](https://img.shields.io/maintenance/yes/2021?style=for-the-badge)
![GitHub repo size](https://img.shields.io/github/repo-size/anirudhtulasi/TwiP?style=for-the-badge)

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/anirudhtulasi/TwiP">
    <img src="https://image.flaticon.com/icons/png/512/60/60580.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">TwiP</h3>

  <p align="center">
    Personality plays a vital role in todays world. Our project will automate the screening process before hiring a professional or can be used in psychiatry to check effectivity of patient therapy as this project identifies the user personality by only taking in the Twitter Username. This uses the Twitter REST API to mine tweets for personality identification. We will use n-grams and word vectors for the hashtags, emoticons and phrases using NLP techniques. We will train the machine to classify the personality types by using a Naive- Bayes Text Classifier and to accurately predict the user’s Myers-Briggs personality type using 10- fold cross validation. We will generate a naive bayes classifier models for all 4 different classes. It will generate few scores which will give the training data size and the features used while training the model.  <br />
<a href="https://github.com/anirudhtulasi/rTwiP/blob/main/Images/MiniProject.docx"><strong>Explore the docs »</strong></a>
 <br />
    <br />
    <a href="https://anirudhxtwitter.herokuapp.com/">View Demo</a>
    ·
    <a href="https://github.com/anirudhtulasi/TwiP/">Report Bug</a>
    ·
    <a href="https://github.com/anirudhtulasi/TwiP/">Request Feature</a>  
    
    
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


We wanted to perform Personality Classification by Suggestion Mining. Unlike other methods we wanted the project to only take the Twitter Username as the input. For this to work we tried to use the Twitter REST API to mine tweets. All the papers only showcased to classify the personality into 4 classes ( Good, Bad, Neutral, N/A) but here we will be using Myers-Briggs Types Indicator and classify into 16 different types.



### Built With

* [pandas]()
* [pickleshare]()
* [numpy]()
* [scikit-learn]()
* [nltk]()
* [tweepy]()
* [unicodecsv]()



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list the packages that you need to run this project.
  
  ```sh
  cd TwiP
  cat requirements.txt
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/anirudhtulasi/TwiP.git
   ```
2. Install required packages
   ```sh
   $ pip install -r requirements.txt
   ```



<!-- USAGE EXAMPLES -->
## Usage
1. Train the Model
   ```sh
   $ cd TwiP
   $ python3 train.py
   ```
2. Run the Model
   ```sh
   $ python3 predict.py
   ```



_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/anirudhtulasi/TwiP/) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Anirudh Tulasi - [@AnirudhTulasi](https://twitter.com/AnirudhTulasi)

Email : [anirudhtulasi.x@gmail.com](mailto:anirudhtulasi.x@gmail.com)

Project Link: [https://github.com/anirudhtulasi/Twip](https://github.com/anirudhtulasi/TwiP)



<!-- ACKNOWLEDGEMENTS 
## Acknowledgements

* []()
* []()
* []()
-->
