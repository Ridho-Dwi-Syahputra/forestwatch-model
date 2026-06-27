# Frontend React + Vite. Build statis lalu serve via `vite preview`.
# VITE_API_URL di-baked saat build (variabel VITE_* bersifat build-time).
#
# Build (dari folder frontend/):
#   docker build --build-arg VITE_API_URL=http://localhost:8000 -t forestwatch-frontend .

FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

EXPOSE 4173

CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]
