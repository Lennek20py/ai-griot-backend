# Digital Griot Backend

An AI-powered platform for preserving and sharing oral traditions, built with FastAPI and Python.

## Overview

The Digital Griot is a cloud-native web platform that allows contributors to upload audio recordings of oral stories. An AI-powered pipeline processes these recordings to provide transcription, translation, and analysis services.

## Features

### Core Features
- **User Authentication**: JWT-based authentication with secure password hashing
- **Story Management**: CRUD operations for oral stories with metadata
- **Audio Upload**: Secure file upload to AWS S3 with validation
- **AI Transcription**: Automatic speech-to-text using OpenAI Whisper and Google Speech-to-Text
- **Multi-language Translation**: AI-powered translation using Google Translate
- **Sentiment Analysis**: Analyze emotional tone of stories
- **Named Entity Recognition**: Extract people, places, and cultural artifacts
- **Search & Filtering**: Advanced search across stories with multiple filters
- **Analytics**: Track views, listens, and engagement metrics

## Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Set up environment**: Copy `.env.example` to `.env` and configure
3. **Run with Docker**: `docker-compose up`
4. **Or run locally**: `uvicorn app.main:app --reload`

## API Documentation

Once running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
app/
├── api/v1/endpoints/     # API endpoints
├── core/                 # Configuration and utilities
├── models/               # Database models
├── tasks/                # Background tasks
└── main.py              # FastAPI application
```

## Tech Stack

- **Framework**: FastAPI 0.104.1
- **Database**: PostgreSQL with asyncpg
- **ORM**: SQLAlchemy 2.0 with async support
- **Authentication**: JWT with python-jose
- **File Storage**: AWS S3 with boto3
- **AI/ML**: OpenAI Whisper, Google Cloud, spaCy, Transformers
- **Background Tasks**: Celery with Redis
