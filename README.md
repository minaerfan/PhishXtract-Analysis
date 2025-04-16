
# PhishXtract_Analysis

Understanding the infrastructure behind phishing websites is crucial for developing effective detection and mitigation strategies. This repository presents a comprehensive analysis of phishing domain ownership by categorizing phishing URLs into three primary types: attacker-owned domains, compromised legitimate domains, and links hosted on third-party platforms. Building on the foundation of the Taxonomy of Phishing Websites project, this extension leverages a machine learning classifier trained on a curated, manually labeled dataset to infer ownership patterns from a large-scale, real-world phishing reports corpus collected over a year.

## Introduction

This is an extension to the Taxonomy of Phishing Websites project. In this work, we categorize phishing websites into three distinct groups: Attacker- Owned Domains, Hosting Platforms, and Compromised Domains. \

## Methodology

This project employs the Random Forest model previously trained on a manually labeled dataset (PhishXtract-class-Labeled) on a phishing reports corpus collected over a year to identify domain ownership patterns of the recent phishing websites

## Getting Started

1. Clone this repository to your local machine.
2. Ensure you have the necessary dependencies installed. Dependencies include:
  - Python3 (The project has been tested on Python 3.12)
  - Scikit-learn
  - pandas
  - numpy
3. Navigate to directories in the cloned repository. 
4. You can find the source code in the 'src' and the dataset for classification in PhishXtract-Class directory
5. Open py file and execute it.
6. Results will be saved in the 'result' directory
7. Analyze the results.


## Contact Information

For questions, please contact [merfa006@uottawa.ca](mailto:merfa006@uottawa.ca).
