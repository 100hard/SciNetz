FROM node:20-alpine
WORKDIR /app
COPY Frontend/package*.json ./
RUN npm install --legacy-peer-deps
COPY Frontend .
EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--port", "3000", "--hostname", "0.0.0.0"]