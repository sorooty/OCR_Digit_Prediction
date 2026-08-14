// An OCR Client (ocr.js)
// CANVAS_WIDTH : canvas reel 200x200 pixels
// TRANSLATED_WIDTH : canvas logique 20x20 = 400 entrees pour le reseau
// PIXEL_WIDTH : 1 pixel logique = carre de 10x10 pixels reels

var ocrDemo = {
  CANVAS_WIDTH: 200,
  TRANSLATED_WIDTH: 20,
  PIXEL_WIDTH: 10,
  BLUE: "#0000ff",
  BLACK: "#000000",
  BATCH_SIZE: 1,
  HOST: "http://localhost",
  PORT: 8000,

  // Tableau de 400 valeurs : 0=pixel vide, 1=pixel dessine
  data: [],
  trainArray: [],
  trainingRequestCount: 0,

  // Appelee par onload dans ocr.html
  onLoadFunction: function () {
    this.resetCanvas();
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = this.BLACK;
    ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
    this.drawGrid(ctx);
    // Listeners souris pour le dessin
    canvas.addEventListener("mousemove", function (e) { ocrDemo.onMouseMove(e, ctx, canvas); });
    canvas.addEventListener("mousedown", function (e) { ocrDemo.onMouseDown(e, ctx, canvas); });
    canvas.addEventListener("mouseup",   function (e) { ocrDemo.onMouseUp(e); });
  },

  // Reinitialise data[] et le canvas (fond noir + grille)
  resetCanvas: function () {
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    // 400 zeros = image vide
    this.data = new Array(this.TRANSLATED_WIDTH * this.TRANSLATED_WIDTH).fill(0);
    ctx.fillStyle = this.BLACK;
    ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
    this.drawGrid(ctx);
  },

  // Trace les lignes de la grille bleue (1 ligne tous les PIXEL_WIDTH px)
  drawGrid: function (ctx) {
    for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH; x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH, y += this.PIXEL_WIDTH) {
      ctx.strokeStyle = this.BLUE;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.CANVAS_WIDTH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.CANVAS_WIDTH, y); ctx.stroke();
    }
  },

  // Dessine seulement si bouton enfonce (isDrawing)
  onMouseMove: function (e, ctx, canvas) {
    if (!canvas.isDrawing) return;
    this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
  },

  // Active le mode dessin
  onMouseDown: function (e, ctx, canvas) {
    canvas.isDrawing = true;
    this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
  },

  // Desactive le mode dessin
  onMouseUp: function (e) { canvas.isDrawing = false; },

  // Colorie le carre 10x10 et met data[index]=1
  // index = (xPixel-1)*20 + (yPixel-1)
  fillSquare: function (ctx, x, y) {
    var xPixel = Math.floor(x / this.PIXEL_WIDTH);
    var yPixel = Math.floor(y / this.PIXEL_WIDTH);
    this.data[(xPixel - 1) * this.TRANSLATED_WIDTH + yPixel - 1] = 1;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, this.PIXEL_WIDTH, this.PIXEL_WIDTH);
  },

  // Accumule BATCH_SIZE exemples puis envoie au serveur
  // trainArray : liste de {y0:[400 valeurs], label:chiffre}
  train: function () {
    var digitVal = document.getElementById("digit").value;
    if (!digitVal || this.data.indexOf(1) < 0) {
      alert("Please type and draw a digit value in order to train the network");
      return;
    }
    this.trainArray.push({ y0: this.data, label: parseInt(digitVal) });
    this.trainingRequestCount++;
    if (this.trainingRequestCount == this.BATCH_SIZE) {
      alert("Sending training data to server...");
      this.sendData({ trainArray: this.trainArray, train: true });
      this.trainingRequestCount = 0;
      this.trainArray = [];
    }
  },

  // Envoie l image et attend la prediction (pas de batch)
  test: function () {
    if (this.data.indexOf(1) < 0) {
      alert("Please draw a digit in order to test the network");
      return;
    }
    this.sendData({ image: this.data, predict: true });
  },

  // Affiche le chiffre predit retourne par le serveur
  receiveResponse: function (xmlHttp) {
    if (xmlHttp.status != 200) { alert("Server returned status " + xmlHttp.status); return; }
    var responseJSON = JSON.parse(xmlHttp.responseText);
    if (xmlHttp.responseText && responseJSON.type == "test") {
      alert("The neural network predicts you wrote a '" + responseJSON.result + "'");
    }
  },

  onError: function (e) { alert("Error occurred while connecting to server: " + e.target.statusText); },

  // HTTP POST vers le serveur Python (requete synchrone)
  sendData: function (json) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open("POST", this.HOST + ":" + this.PORT, false);
    xmlHttp.onload = function () { this.receiveResponse(xmlHttp); }.bind(this);
    xmlHttp.onerror = function () { this.onError(xmlHttp); }.bind(this);
    var msg = JSON.stringify(json);
    xmlHttp.setRequestHeader("Content-length", msg.length);
    xmlHttp.setRequestHeader("Connection", "close");
    xmlHttp.send(msg);
  },
};
