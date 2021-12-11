"use strict";

const routeResponseMap = {
  "/info" : "<h1>Info Page</h1>",
  "/contact" : "<h1>Concat Us</h1>",
  "/about" : "<h1>Lean More About Us</h1>",
  "/hello": "<h1>Say hello by emailing us here</h1>",
  "/error": "<h1>Sorry, the page you are looking for is not here</h1>"
}

const port = 3000,
      http = require("http"),
      httpStatus = require("http-status-codes"),
      app = http.createServer();

app.on("request", (req, res) => {
  var body = [];
  req.on("data", (bodyData) => {
    body.push(bodyData)
  });
  req.on("end", () => {
    body = Buffer.concat(body).toString();
    console.log(`Request Body Contesnt: ${body}`);
  });
  
  console.log(req.method);
  console.log(req.url);
  console.log(req.headers);

  res.writeHead(httpStatus.OK, {
    "Content-Type": "text/html"
  });
  if (routeResponseMap[req.url]) {
    res.end(routeResponseMap[req.url]);
  } else {
    res.end("<h1>Wellcome</h1>");
  }
});


app.listen(port);
console.log(`The server has started and is listening on port number :${port}`);
