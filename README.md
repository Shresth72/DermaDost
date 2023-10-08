# DermaDost - Skin Condition Detector Web App
DermaDost is a prototype web application designed to help users identify various skin conditions, provide detailed information about these conditions through articles from the internet, and offer a chatbot feature for discussing skin-related concerns. Additionally, it allows users to capture images through their webcam and analyze them for potential skin conditions.

## Hosted here: [DermaDost-Skin-Disease](https://skin-disease.yasharya5.repl.co)
<img src="https://github.com/Shresth72/DermaDost/assets/97455610/0a562c6f-1fd1-4a06-b0d0-d211ae86a0b6" alt="mockup" width="900"/>

#### We are team BACKDOOR INNOVATORS for Smart India Hackathon
### Team Members:
- Ashwary Tripathi (Leader)
- Shrestha Shashank
- Ahbar Husain
- Amal Kumar
- Indrani Dutta
- Yash Arya

## Video Links of our submissions
### SIH Project Id: 1344
#### PPT LINKS: 
##### https://drive.google.com/file/d/1FaGeMgI2m0IaAEgXnKgmuPFvjToRGy2P/view?usp=drivesdk
##### https://docs.google.com/presentation/d/1D5xysT-OU6X3faNdOQWFPwg_kD7yw4ZrjJPNRtctB_M/edit?usp=drivesdk


### Video Presentation of this prototype
#### Video Link: https://drive.google.com/file/d/1FTg8-zFosntr5ZhEibUZen9Kpr-XqKt8/view?usp=drivesdk


# How It Works
DermaDost combines multiple features to assist users in understanding and managing their skin conditions:
- **Upload Images:** Users can upload images of their skin concerns. The application then uses machine learning models to identify potential skin conditions in the images.

- **Information Articles:** Once a condition is identified, DermaDost provides relevant articles from the internet. These articles offer detailed information about the skin condition, its causes, symptoms, and possible treatments.

- **Chatbot Support:** One of the standout features is the ability for users to engage with a chatbot to discuss their skin concerns further. The chatbot can provide general advice and answer common questions about skin health.

- **Authentication:** To ensure the security and privacy of your invoices, the app includes user authentication. This means you can control who has access to your invoices and other app features.

- **Webcam Capture:** DermaDost also allows users to capture images directly through their webcam. These images can be analyzed for potential skin conditions just like uploaded images.

- **JWT Authentication**: JSON Web Tokens (JWT) are used for secure user authentication. JWTs enable safe transmission of data between the app and the Firebase backend, ensuring that your data remains confidential.

## Prerequisites
Prerequisites
Before you begin, ensure you have met the following requirements:

- **Python:** DermaDost is built using Python, so make sure you have Python installed on your system. You can download it from python.org or use a package manager like Anaconda if you prefer.

- **Webcam Access (Optional):** If you want to use the webcam capture feature, ensure that your computer has a webcam, and you have granted permission for the browser to access it.

- **Git (Optional):** To clone the repository and contribute, you'll need Git installed on your system. Download it from git-scm.com.

- **Internet Connection:** DermaDost fetches information articles from the internet, so an active internet connection is required.

## Installation
To run DermaDost on your system, follow these steps:

1. Download the app's source code by cloning this repository to your local machine using the following command:

```bash
git clone https://github.com/yourusername/DermaDost.git
cd DermaDost
```

2. Install the required packages from requirements.txt using pip:
 ```bash
 py -m pip install -r requirements.txt
 ```
    
3. Run the application:
 ```bash
 python app.py
```
This will start the application locally. You can access it in your web browser at http://localhost:5000.

## Usage
- Open your web browser and go to http://localhost:5000.
- Upload an image of your skin concern or capture one through your webcam.
- Wait for the application to identify the skin condition.
- Explore detailed information about the condition and possible treatments through provided articles.
- Use the chatbot feature to discuss your skin concerns further.


### Enjoy using DermaDost! If you have any questions or encounter any issues, please feel free to create an issue on our GitHub repository.
