
sed -i '' "s~dir_path =.*~dir_path = \"`pwd`\"~g" ./src/analyzer_menu.py
sed -i '' "s~string DIR_PATH =.*~string DIR_PATH = \"`pwd`\";~g" ./src/golf_analyzer.cpp

rm -rf build_golf
mkdir build_golf
cd build_golf
cmake ../src
make
cd ..
pyinstaller --onefile -w ./src/analyzer_menu.py
mv ./dist/analyzer_menu ./
rm -rf build dist analyzer_menu.spec
mv analyzer_menu analyzer.app
brew install fileicon
fileicon set analyzer.app ./data/icon.png