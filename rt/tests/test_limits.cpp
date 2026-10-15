// Property checks on the C++ limiter (mirrors the Python safety tests).
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "roboemot/motion.hpp"

using namespace roboemot;

static int bad = 0;
#define CHECK(cond, msg)                         \
  do {                                           \
    if (!(cond)) {                               \
      std::fprintf(stderr, "FAIL: %s\n", msg);   \
      ++bad;                                     \
    }                                            \
  } while (0)

static void velocity_acceleration_no_overshoot() {
  Dof d{"j", -1.0, 1.0, 0.8, 3.0, 0.0};
  Limiter lim({d}, {});
  const double dt = 0.02;
  double prev_x = 0.0, prev_v = 0.0, peak = -1e9;
  for (int i = 0; i < 400; ++i) {
    const double x = lim.step({{"j", 5.0}}, dt).at("j");
    const double v = (x - prev_x) / dt;
    CHECK(std::fabs(v) <= d.max_vel + 1e-9, "velocity cap");
    CHECK(std::fabs(v - prev_v) / dt <= d.max_acc + 1e-6, "acceleration cap");
    peak = std::max(peak, x);
    prev_x = x;
    prev_v = v;
  }
  CHECK(peak <= 1.0 + 1e-12, "no overshoot past max");
  CHECK(std::fabs(lim.position().at("j") - 1.0) < 1e-9, "reaches the clamped goal");
}

static void factor_drop_mid_motion_stays_in_limits() {
  // Review finding 2: a minor appears while the joint rushes toward its limit.
  Dof d{"j", -1.0, 1.0, 2.0, 4.0, 0.0};
  Limiter lim({d}, {});
  const double dt = 0.01;
  double x = 0.0;
  for (int i = 0; i < 60; ++i) x = lim.step({{"j", 1.0}}, dt, 1.0).at("j");
  for (int i = 0; i < 300; ++i) {
    x = lim.step({{"j", 0.6 * 1.0}}, dt, 0.3).at("j");
    CHECK(x <= 1.0 + 1e-12, "position limit with factor drop");
  }
}

static void rejects_bad_inputs() {
  Dof d{"j", -1.0, 1.0, 1.0, 1.0, 0.0};
  Limiter lim({d}, {});
  bool threw = false;
  try { lim.step({{"j", std::numeric_limits<double>::quiet_NaN()}}, 0.01); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw, "NaN target rejected");
  threw = false;
  try { lim.step({{"j", 0.1}}, 0.0); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw, "dt=0 rejected");
  threw = false;
  try { lim.step({{"j", 0.1}}, 0.01, 0.0); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw, "factor=0 rejected");
  CHECK(std::isfinite(lim.position().at("j")), "state stays finite");
}

static void constraint_coupling_respects_velocity() {
  // Review finding 6: body runs away faster than the head can follow.
  Dof head{"head.yaw", -3.0, 3.0, 0.5, 5.0, 0.0}, body{"body_yaw", -2.8, 2.8, 2.0, 20.0, 0.0};
  const double lim65 = 65.0 * M_PI / 180.0;
  Limiter lim({head, body}, {{"head.yaw", "body_yaw", lim65}}, 3.0 * M_PI / 180.0);
  const double dt = 0.01;
  double ph = 0.0, pb = 0.0, vh = 0.0, vb = 0.0;
  for (int i = 0; i < 300; ++i) {
    const Values out = lim.step({{"head.yaw", 0.0}, {"body_yaw", 2.5}}, dt);
    const double nvh = (out.at("head.yaw") - ph) / dt, nvb = (out.at("body_yaw") - pb) / dt;
    CHECK(std::fabs(nvh) <= head.max_vel + 1e-9, "head velocity");
    CHECK(std::fabs(nvb) <= body.max_vel + 1e-9, "body velocity");
    CHECK(std::fabs(nvh - vh) / dt <= head.max_acc + 1e-6, "head acceleration");
    CHECK(std::fabs(nvb - vb) / dt <= body.max_acc + 1e-6, "body acceleration");
    CHECK(std::fabs(out.at("head.yaw") - out.at("body_yaw")) <= lim65 + 1e-9, "65 deg kept");
    ph = out.at("head.yaw");
    pb = out.at("body_yaw");
    vh = nvh;
    vb = nvb;
  }
}

int main() {
  velocity_acceleration_no_overshoot();
  factor_drop_mid_motion_stays_in_limits();
  rejects_bad_inputs();
  constraint_coupling_respects_velocity();
  std::printf("limits: violations = %d\n", bad);
  return bad == 0 ? 0 : 1;
}
