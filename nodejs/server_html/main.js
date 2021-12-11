"use strict";

const port = 3000,
      http = require("http"),
      httpStatus = require("http-status-codes"),
      fs = require("fs"),
      router = require("./router"),
      plainTextContentType = {
        "Content-Type": "text/plain"
      },
      htmlContentType = {
        "Content-Type": "text/html"
      };

const customReadFile = (file_path, res) => {
  if (fs.existsSync(file_path)) {
    fs.readFile(file_path, (error, data) => {
      if (error) {
        console.log(error);
        sendErrorResponse(res);
        return;
      }
      res.write(data);
      res.end();
    });
  } else {
    sendErrorResponse(res);
  }
}

router.get("/", (req, res) => {
  res.writeHead(httpStatus.OK, plainTextContentType);
  res.end("INDEX");
});

router.get("/index.html", (req, res) => {
  res.writeHead(httpStatus.OK, htmlContentType);
  customReadFile("views/index.html", res);
})

router.post("/", (req, res) => {
  res.writeHead(httpStatus.OK, plainTextContetType);
  res.end("POSTED");
})

http.createServer(router.handle).listen(port);

console.log(`The server has started and is listening on port number ${port}`);


