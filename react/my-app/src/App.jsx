import { useState, useEffect } from "react";
import { ColoredMessage } from "./components/ColoredMessage";
import { CssModule } from "./components/CssModule";

export const App = () => {
  const [num, setNum] = useState(0);
  useEffect(() => {
    alert();
  }, [num]);

  const onClickButton = () => {
    setNum((prev) => prev + 1);
    setNum((prev) => prev + 1);
  };

  return (
    <div>
      <h1 style={{ color: "red" }}>Hello World!</h1>
      <ColoredMessage color="blue">"How are you?"</ColoredMessage>
      <ColoredMessage color="pink">"I'm fine!"</ColoredMessage>
      <button onClick={onClickButton}>button</button>
      <p>{num}</p>
      <CssModule />
    </div>
  );
};
