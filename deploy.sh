#!/bin/bash
set -e

PROJECT_ID="afs-jasoncrites-1766633078257"
SERVICE_NAME="serverless-supercomputer"
REGION="us-central1"

echo "🚀 Deploying $SERVICE_NAME to $PROJECT_ID..."

# Build using Cloud Build
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$SERVICE_NAME . --project $PROJECT_ID

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$SERVICE_NAME \
  --ingress all \
  --platform managed \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
  --set-env-vars="GOOGLE_API_KEY=$GOOGLE_API_KEY"

echo "✅ Success! API live at: $(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)' --project $PROJECT_ID)"
