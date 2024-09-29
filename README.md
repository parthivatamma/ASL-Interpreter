# ASL-Interpreter

## Inspiration
We have had deaf friends who had difficulty communicating with others who don't know ASL. So we set out to develop an ASL interpreter that could provide real-time, accurate translations in everyday situations. It isn't just about technology; it's about empowering our friends and countless others in the deaf community to navigate the world with greater ease and confidence. 

## What it does
Our program utilizes real-time video data, detects for hands, and interprets the symbol the user is trying to display.

## How we built it
Using Mediapipe's hand landmarks detection library, we used a dataset we found online (https://datasets.cms.waikato.ac.nz/ufdl/american-sign-language-letters/) and trained a machine learning model with open-cv to recognize different hand signs. 
We first created and tested a quick program to see if we could get the libraries running and recognizing hands through video. We then created a dataset using the data that we already had. After, we trained and tested the machine learning model and then implemented it into a simple program that could guess what symbols users were putting up in real-time.

## Challenges we ran into
One of our first and early challenges was finding a dataset. There were many online, and our first idea was to get a large dataset that had almost 82,000 images to train our model with. However, we realized that the time it took to get through the data was too long and too strenuous on our machines. We then went with a dataset that was a lot smaller but would be a lot easier to debug and get through without sacrificing too much accuracy. Our next issue was that the data had several inconsistencies so our program couldn't read the dataset properly and kept crashing. To solve this issue we padded the data so that all the data in the data set had 84 features.

## Accomplishments that we're proud of
We learned how to train and create a machine-learning model. We learned how to use our computer science knowledge to tackle a real-world problem that can help real people.

## What we learned
We learned how to use Pickle to store a dataset within Python. We learned and gained a better understanding of how to use Open CV and Mediapipe to train and test a machine-learning model.

## What's next for American Sign Language Interpreter
One of our goals now is to try and refine the model, as its accuracy isn't at 100% yet. Perhaps creating a dataset would help us. We also would like to create an easy-to-use UI, or maybe even implement this into a website or a mobile device application. Eventually, we plan on adding a module to store the characters into strings and then adding a text-to-speech function to be able to speak the characters.
