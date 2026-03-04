#!/bin/bash
#
# GLM-OCR: Tạo VM trên Google Compute Engine với L4 GPU
#
# Cách dùng:
#   1. Cài Google Cloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Đăng nhập: gcloud auth login
#   3. Chạy script này: bash setup_gcloud.sh
#
set -e

# ============================
# CẤU HÌNH — CHỈNH SỬA Ở ĐÂY
# ============================
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"     # ← Đổi thành project ID của bạn
ZONE="asia-northeast1-a"                             # Tokyo (gần VN), có L4
VM_NAME="glm-ocr-server"
MACHINE_TYPE="g2-standard-4"                          # 4 vCPU, 16GB RAM, 1x L4 GPU
DISK_SIZE="100"                                       # GB
APP_PASSWORD="${APP_PASSWORD:-glm-ocr-2024}"          # ← Đổi mật khẩu
# ============================

echo "═══════════════════════════════════════════"
echo "  GLM-OCR — Google Cloud VM Setup"
echo "═══════════════════════════════════════════"
echo ""
echo "  Project:  $PROJECT_ID"
echo "  Zone:     $ZONE"
echo "  VM:       $VM_NAME"
echo "  Machine:  $MACHINE_TYPE (L4 GPU)"
echo "  Password: $APP_PASSWORD"
echo ""

# Set project
gcloud config set project "$PROJECT_ID"

# Enable APIs
echo "📦 Enabling Compute Engine API..."
gcloud services enable compute.googleapis.com

# Create firewall rule for port 5000
echo "🔥 Creating firewall rule..."
gcloud compute firewall-rules create allow-glm-ocr \
    --allow=tcp:5000 \
    --target-tags=glm-ocr \
    --description="Allow GLM-OCR web app" \
    --direction=INGRESS \
    2>/dev/null || echo "   (Firewall rule already exists)"

# Create VM
echo ""
echo "🚀 Creating VM with L4 GPU..."
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator=type=nvidia-l4,count=1 \
    --boot-disk-size="${DISK_SIZE}GB" \
    --boot-disk-type=pd-ssd \
    --image-family=common-cu124-debian-12 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --tags=glm-ocr \
    --metadata=startup-script="#!/bin/bash
# Wait for GPU drivers
nvidia-smi || sleep 30

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Install NVIDIA Container Toolkit
if ! command -v nvidia-ctk &> /dev/null; then
    distribution=\$(. /etc/os-release; echo \$ID\$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# Create data directory
mkdir -p /data/input_pdfs /data/output_markdown

# Build and run the app
cd /opt/glm-ocr 2>/dev/null || true
if [ -f Dockerfile ]; then
    docker build -t glm-ocr .
    docker rm -f glm-ocr 2>/dev/null || true
    docker run -d --name glm-ocr \
        --gpus all \
        -p 5000:5000 \
        -v /data:/data \
        -e APP_PASSWORD='${APP_PASSWORD}' \
        --restart unless-stopped \
        glm-ocr
fi
"

echo ""
echo "⏳ Waiting for VM to start..."
sleep 10

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$ZONE" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ VM Created!"
echo "═══════════════════════════════════════════"
echo ""
echo "  IP:       $EXTERNAL_IP"
echo "  URL:      http://$EXTERNAL_IP:5000"
echo "  Password: $APP_PASSWORD"
echo ""
echo "═══ TIẾP THEO ═══"
echo ""
echo "  1. Upload deploy files vào VM:"
echo "     gcloud compute scp deploy/* $VM_NAME:/opt/glm-ocr/ --zone=$ZONE"
echo ""
echo "  2. SSH vào VM để build Docker:"
echo "     gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "     cd /opt/glm-ocr && sudo docker build -t glm-ocr ."
echo "     sudo docker run -d --name glm-ocr --gpus all -p 5000:5000 -v /data:/data -e APP_PASSWORD='$APP_PASSWORD' --restart unless-stopped glm-ocr"
echo ""
echo "  3. Truy cập: http://$EXTERNAL_IP:5000"
echo ""
echo "═══ QUẢN LÝ VM ═══"
echo ""
echo "  Tắt VM (tiết kiệm):  gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo "  Bật VM:               gcloud compute instances start $VM_NAME --zone=$ZONE"
echo "  Xóa VM:               gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
echo "  💰 Chi phí: ~\$0.95/giờ khi chạy | ~\$17/tháng disk khi tắt"
echo ""
