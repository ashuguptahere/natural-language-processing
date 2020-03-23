# Importing Essential Libraries
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import pandas as pd
import sys
import re

# Importing NLTK
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# Creating a class
class NLP_reviews_GUI(QDialog):

    # Defining INIT Function
    def __init__(self):
        QDialog.__init__(self)
        layout = QVBoxLayout()
        self.setWindowTitle("Review Classifier")
        self.progress = QProgressBar()
        cbutton = QPushButton("Close")
        browse = QPushButton("Browse for dataset")
        train = QPushButton("Train")
        scrape = QPushButton("Scrape")
        fit_score = QPushButton("GNB Fit Score")
        pred = QPushButton("Predict")

        self.progress.setValue(0)
        self.progress.setAlignment(Qt.AlignHCenter)

        self.loc = QLineEdit(self)
        self.loc.setPlaceholderText("Your File's Location...")

        self.loc1 = QLineEdit(self)
        self.loc1.setPlaceholderText("Write any review and click predict...")

        self.loc2 = QLineEdit(self)
        self.loc2.setPlaceholderText('Please provide the link(s) that you want to scrape or you can browse a file...')

        cbutton.clicked.connect(self.close)
        layout.addWidget(self.loc2)
        layout.addWidget(scrape)
        layout.addWidget(self.loc)
        layout.addWidget(browse)
        layout.addWidget(self.progress)
        layout.addWidget(train)
        layout.addWidget(fit_score)
        layout.addWidget(self.loc1)
        layout.addWidget(pred)
        layout.addWidget(cbutton)

        self.loc.move(80, 20)
        self.loc.resize(200, 32)
        self.setLayout(layout)
        self.setFocus()

        browse.clicked.connect(self.browse_file)
        train.clicked.connect(self.train)
        fit_score.clicked.connect(self.scorefn)
        pred.clicked.connect(self.prediction)
        scrape.clicked.connect(self.scrapefn)

    def links(self):
        try:
            return str(self.loc2.text())
        except Exception:
            QMessageBox.information(self, "Error", "Error 404...".format(QMessageBox.warning))

    def scrapefn(self):
        self.scrape = dialog.links()
        # Coping all Links into variable 'links'
        try:
            #with open('Links.csv') as f:
            #    reader = csv.reader(f)
            #    links = [row[0] for row in reader]

            print(self.links)
            print(type(self.links))

            from selenium import webdriver
            driver = webdriver.Chrome(executable_path = 'C:\chromedriver_win32\chromedriver.exe')

            reviews = []
            for link in links:
                response = driver.get(link)
                html_source = driver.page_source
                soup = BeautifulSoup(html_source, 'html.parser')
                i = soup.find_all('div', {'class': 'reviewSnippetCard'})
                for j in i:
                    review = j.find('p', {'class': 'snippetSummary'}).text
                    rating = int(j.find('span', {'class': 'stars-icon-star-on'})['style'][6:-2]) / 20
                    reviews.append([review, rating])

            dataset = pd.DataFrame(reviews, columns=['Review', 'Rating'])
            dataset.to_csv('reviews.csv', index=False)
        except Exception:
            QMessageBox.information(self, "Error", "There is something wrong with the entered links...".format(QMessageBox.warning))

    # Function for browsing a file
    def browse_file(self):
        try:
            self.name = QFileDialog.getOpenFileName(self, caption='Open File', filter="All Files (*.*)")
            self.loc.setText(self.name)
        except Exception:
            QMessageBox.information(self, "Predict", "Unable to Open the File...".format(QMessageBox.warning))

    # Function for training the dataset
    def train(self):
        try:
            dataset = pd.read_csv(self.name)
            self.processed_reviews = []

            # Preprocessing the data for NLP
            for i in range(len(dataset)):
                review = re.sub('@[\w]*', ' ', dataset['Review'][i])
                review = re.sub('^a-zA-Z#', ' ', review)
                review = review.lower()
                review = review.split()
                review = [ps.stem(token) for token in review if not token in stopwords.words('english')]
                review = ' '.join(review)
                self.processed_reviews.append(review)

                if i > len(dataset) / 100:
                    percent = int((i * 100) / len(dataset)) + 2
                    self.progress.setValue(percent)

            from sklearn.feature_extraction.text import CountVectorizer
            self.cv = CountVectorizer(max_features = 3000)

            # Defining X (Feature Matrix) and y (Vector of Predictions)
            self.X = self.cv.fit_transform(self.processed_reviews).toarray()
            y = dataset['Rating'].values

            # Making y in binary(either 0 or 1)
            for i in range(len(y)):
                if y[i] >= 4:
                    y[i] = 1  # 1 means Positive
                else:
                    y[i] = 0  # 0 means Negative

            from sklearn.naive_bayes import GaussianNB
            self.n_b = GaussianNB()
            self.n_b.fit(self.X, y)
            self.score1 = float(self.n_b.score(self.X, y) * 100)
            QMessageBox.information(self, "Training Error", "Your Model is Trained... Press OK".format(QMessageBox.Ok))
        except Exception:
            QMessageBox.information(self, "Training Error", "Unable to Train the Naive Bayes Model...".format(QMessageBox.warning))

    def scorefn(self):
        try:
            QMessageBox.information(self, "Training Error", "Naive Bayes Score is {}{}".format(self.score1, '%'))
        except Exception:
            QMessageBox.information(self, "Training Error", "Unable to Calculate Naive Bayes Score...".format(QMessageBox.warning))

    def review(self):
        try:
            return self.loc1.text()
        except Exception:
            QMessageBox.information(self, "Prediction Error", "Error 404...".format(QMessageBox.warning))

    def prediction(self):
        try:
            self.z = dialog.review()
            review1 = re.sub('@[\w]*', ' ', self.z)
            review1 = re.sub('^a-zA-Z#', ' ', review1)
            review1 = review1.lower()
            review1 = review1.split()
            review1 = [ps.stem(token) for token in review1 if not token in stopwords.words('english')]
            review1 = ' '.join(review1)
            self.processed_reviews.append(review1)

            from sklearn.feature_extraction.text import CountVectorizer
            self.cv = CountVectorizer(max_features = len(self.X.transpose()))
            X1 = self.cv.fit_transform(self.processed_reviews).toarray()

            if self.n_b.predict(X1[[-1]]) == 1:
                QMessageBox.information(self, "Predict", "Rating is Positive")
            else:
                QMessageBox.information(self, "Predict", "Rating is Negative")
        except Exception:
            QMessageBox.information(self, "Predict", "Error 404...".format(QMessageBox.warning))


app = QApplication(sys.argv)
dialog = NLP_reviews_GUI()
dialog.show()
app.exec_()
