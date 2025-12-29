.PHONY : clean data

output/train.txt : train.py data/ml-1m.db
	python $< data/ml-1m.db

# Download the stable 1m dataset from the MovieLens website.

data : data/ml-1m/README

data/ml-1m.zip :
	@echo "⬇️ Downloading MovieLens 1m dataset ..."
	mkdir -p data/
	curl -L -o $@ https://files.grouplens.org/datasets/movielens/ml-1m.zip
	@echo "✅ Downloaded MovieLens 1m dataset."

data/ml-1m/README : data/ml-1m.zip
	@echo "📦 Extracting MovieLens 1m dataset ..."
	unzip -o -d data $<
	# Touch the output to update the timestamp.
	touch $@
	@echo "✅ Extracted MovieLens 1m dataset."

data/ml-1m.db : build_database.py data/ml-1m/README
	@echo "🛠️ Building MovieLens 1m SQLite database ..."
	python $< data/ml-1m/ $@
	@echo "✅ Built MovieLens 1m SQLite database."

# Other targets.

clean :
	rm -rf data/
