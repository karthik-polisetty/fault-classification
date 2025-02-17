```markdown
# 🚀 MMS Project - FastAPI Deployment on AWS EC2

This repository contains the **Fault Classification API** for rotating machines, built with **FastAPI** and deployed on **AWS EC2**.

## **📌 Connecting to the AWS EC2 Instance**
To access the EC2 instance, open a terminal and run:

```bash
ssh -i "C:/Users/kpoli/OneDrive/Desktop/mms/fault-classification/data/aws/mms_fastapi.pem" ec2-user@<public_ip>
```
📌 Replace `<public_ip>` with your actual EC2 instance's public IP.

## **📌 Navigate to the Project Directory**
Once connected to the EC2 instance, go to the project folder:

```bash
cd fault-classification
```

## **📌 Pull the Latest Changes from GitHub**
If you've made updates locally, pull the latest changes:

```bash
git pull origin main
```

📌 Ensure you're on the correct branch before pulling changes:
```bash
git branch  # Check current branch
git checkout main  # Switch to main branch if needed
```

## **📌 Activate the Virtual Environment**
Before running the application, activate the Python virtual environment:

```bash
source venv/bin/activate
```
✅ You should now see the `(venv)` prefix in your terminal.

## **📌 Running the FastAPI Application**
Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

🔹 The API will be accessible at:
```
http://<public_ip>:8000/docs
```
or if running behind NGINX:
```
http://<public_ip>/
```
```