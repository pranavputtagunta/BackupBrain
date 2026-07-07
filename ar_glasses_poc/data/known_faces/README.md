# Known faces database

One subdirectory per person, containing one or more labeled photos:

```
data/known_faces/
├── john/
│   └── photo1.jpg
└── sarah/
    └── photo1.jpg
```

Real face photos are not committed to the repo. To enroll someone, run:

```
python setup_faces.py john path/to/john.jpg
```

Sample memory files for `john` and `sarah` already exist in
`data/memories/`, so as soon as you add a photo for either name the full
RAG pipeline works end-to-end. With no photos enrolled, the pipeline
still runs — every detected face is labeled "Unknown".
