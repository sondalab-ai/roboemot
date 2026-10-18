// Turns on the kit's effects layer (aurora, press, glitch). Off in high contrast and under
// reduced motion; the page is complete without it. Views render later: mount() watches for them.
import { burst, mount } from "./vendor/sondalab-ui-kit/effects.js";

mount();
window.roboemotFx = { burst };
