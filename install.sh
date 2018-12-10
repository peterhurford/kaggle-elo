# Remember to scp files and kaggle.json first
sudo apt update
sudo apt install -y python3-pip
sudo apt install -y python-pip
sudo apt install -y unzip
sudo apt install -y htop
sudo apt install -y libsm6 libxext6

pip3 install Cython
pip3 install -r requirements.txt

kaggle competitions download -c elo-merchant-category-recommendation -p .
rm Data_Dictionary.xlsx
unzip train.csv
unzip test.csv
unzip new_merchant_transactions.csv
unzip merchants.csv
unzip historical_transactions.csv.zip
rm *.zip
mkdir data
mkdir cache
mkdir submit
mv *.csv data/.
sudo chown ubuntu -R ~/data
sudo chmod -R 777 ~/data
sudo chown ubuntu -R ~/cache
sudo chmod -R 777 ~/cache
sudo chown ubuntu -R ~/submit
sudo chmod -R 777 ~/submit
