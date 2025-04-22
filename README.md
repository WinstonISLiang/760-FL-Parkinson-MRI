<h3 align="center">Privacy-Preserving Federated Learning for Early Parkinson's Detection through Decentralized MRI Analysis</h3>

<div align="center">

* ignore below will edit links at some point 

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Platform](https://img.shields.io/badge/platform-reddit-orange.svg)](https://www.reddit.com/user/Wordbook_Bot)
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> 🤖 yes yes will change later
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Preprocessing](#demo)
- [Methodology](#working)
- [Usage](#usage)
- [Results](#results)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Built Using](#built_using)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

Write about 1-2 paragraphs describing the purpose of your project.

## 🎥 Preprocessing <a name = "demo"></a>

TODO

## 💭 Methodology <a name = "working"></a>

TODO

## 🎈 Usage <a name = "usage"></a>

TODO

### Example:

TODO

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

Download the NTUA Parkinson Dataset 

```
git clone https://github.com/ails-lab/ntua-parkinson-dataset.git
```

Navigate to the parent directory where both folders are located, and run the following commands:

```
mv "Non PD Patients" "non-pd-patients"
mv "PD Patients" "pd-patients"
```

## 🚀 Deployment <a name = "deployment"></a>

After downloading the dataset, below installs the required dependencies.

```
conda env create -f environment.yml
conda activate sitk-env
```

(Optional) Register the kernel for Jupyter

If you're working in Jupyter notebooks and want to select this environment as a kernel:

```
python -m ipykernel install --user --name=sitk-env
```

### 🙋 Results

TODO


## ⛏️ Built Using <a name = "built_using"></a>

TODO


## ✍️ Authors <a name = "authors"></a>

- [@mchhour](https://github.com/subtlewow) - Idea & Initial work

See also the list of [contributors](https://github.com/subtlewow/760-FL-Parkinson-MRI/graphs/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References
