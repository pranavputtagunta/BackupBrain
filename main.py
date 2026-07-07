import cv2
import numpy as np
import face_recognition
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="BackupBrain Vision Worker")