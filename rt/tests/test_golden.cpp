// Replays golden vectors produced by scripts/gen_golden.py from the Python reference and
// checks that the C++ core produces the same outputs (|diff| <= 1e-9).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "roboemot/motion.hpp"

using namespace roboemot;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: test_golden <golden_motion.txt>\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  std::vector<Dof> dofs;
  std::vector<YawDelta> yd;
  std::unique_ptr<MotionCore> core;
  double dt = 0, now = 0, max_err = 0;
  int ticks = 0, failures = 0;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string kind;
    ss >> kind;
    if (kind == "dof") {
      Dof d;
      ss >> d.name >> d.min >> d.max >> d.max_vel >> d.max_acc >> d.home;
      dofs.push_back(d);
    } else if (kind == "yaw_delta") {
      YawDelta c;
      ss >> c.a >> c.b >> c.max;
      yd.push_back(c);
    } else if (kind == "config") {
      double cf, wd, wr, margin;
      ss >> cf >> wd >> wr >> margin >> dt;
      core = std::make_unique<MotionCore>(dofs, yd, cf, wd, wr, margin);
    } else if (kind == "tick") {
      ss >> now;
    } else if (kind == "cmd") {
      Command c;
      int prio, add, n;
      double exp;
      ss >> c.source >> prio >> add >> c.stamp >> exp >> n;
      c.priority = static_cast<Priority>(prio);
      c.additive = add != 0;
      if (exp >= 0) c.expires = exp;
      for (int i = 0; i < n; ++i) {
        std::string k;
        double v;
        ss >> k >> v;
        c.targets[k] = v;
      }
      core->feed(c);
    } else if (kind == "expect") {
      Values got = core->tick(now, dt);
      int n;
      ss >> n;
      for (int i = 0; i < n; ++i) {
        std::string k;
        double v;
        ss >> k >> v;
        const double err = std::fabs(got[k] - v);
        max_err = std::max(max_err, err);
        if (err > 1e-9 && failures++ < 10)
          std::fprintf(stderr, "t=%.3f %s: got %.12f want %.12f\n", now, k.c_str(), got[k], v);
      }
      ++ticks;
    }
  }
  std::printf("golden: %d ticks, max |err| = %.3g, failures = %d\n", ticks, max_err, failures);
  return failures == 0 && ticks > 0 ? 0 : 1;
}
