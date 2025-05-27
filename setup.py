from setuptools import setup, find_packages

setup(
    name="pneumonia_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'matplotlib',
        'scikit-image',
        'scikit-learn',
        'seaborn',
        'kagglehub',
        'streamlit',
        'pymongo',
        'pytest',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A pneumonia detection system using chest X-ray images with MongoDB integration",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/pneumonia_detection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)