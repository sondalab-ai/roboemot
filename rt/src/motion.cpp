// See include/roboemot/motion.hpp and the Python reference roboemot/motion/core.py.
#include "roboemot/motion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace roboemot {

namespace curves {
namespace {
double clamp01(double s) { return s <= 0.0 ? 0.0 : (s >= 1.0 ? 1.0 : s); }
constexpr double kCartoonOvershoot = 1.70158;
}  // namespace
double linear(double s) { return clamp01(s); }
double min_jerk(double s) {
  s = clamp01(s);
  return s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
}
double ease_in_out(double s) {
  s = clamp01(s);
  return 3.0 * s * s - 2.0 * s * s * s;
}
double cartoon(double s) {
  s = clamp01(s) - 1.0;
  const double c = kCartoonOvershoot;
  return 1.0 + (c + 1.0) * s * s * s + c * s * s;
}
}  // namespace curves

namespace {
constexpr int kStopSearchIterations = 40;
constexpr double kLandTolerance = 1.0 + 1e-9;
}

double stop_distance(double v, double adt, double dt) {
  const double n = std::floor(v / adt);
  return dt * ((n + 1.0) * v - adt * n * (n + 1.0) / 2.0);
}

double stop_speed(double distance, double amax, double dt) {
  if (amax <= 0.0 || dt <= 0.0 || distance <= 0.0) return 0.0;
  const double adt = amax * dt;
  double lo = 0.0, hi = adt * (std::sqrt(2.0 * distance / (adt * dt) + 0.25) + 0.5);
  for (int i = 0; i < kStopSearchIterations; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (stop_distance(mid, adt, dt) <= distance) lo = mid; else hi = mid;
  }
  return lo;
}

// --------------------------------------------------------------------------- Arbiter

Arbiter::Arbiter(std::vector<Dof> dofs, double crossfade_s)
    : dofs_(std::move(dofs)), crossfade_s_(crossfade_s) {
  for (const auto& d : dofs_) {
    winner_[d.name] = std::nullopt;
    fade_[d.name] = std::nullopt;
    last_[d.name] = d.home;
  }
}

namespace {
void require_finite(const Values& values) {
  for (const auto& [name, v] : values)
    if (!std::isfinite(v)) throw std::invalid_argument("non-finite target for " + name);
}
}  // namespace

void Arbiter::submit(const Command& cmd) {
  for (const auto& [name, _] : cmd.targets)
    if (!last_.count(name)) throw std::out_of_range("unknown dof " + name);
  require_finite(cmd.targets);
  if (!std::isfinite(cmd.stamp)) throw std::invalid_argument("non-finite command stamp");
  cmds_[cmd.source] = cmd;
}

void Arbiter::clear(const std::string& source) { cmds_.erase(source); }

Values Arbiter::step(double now, double amplitude) {
  for (auto it = cmds_.begin(); it != cmds_.end();) {
    if (it->second.expires && *it->second.expires <= now) it = cmds_.erase(it); else ++it;
  }
  Values out;
  for (const auto& dof : dofs_) {
    const Command* best = nullptr;
    for (const auto& [_, c] : cmds_) {
      if (c.additive || !c.targets.count(dof.name)) continue;
      if (!best || static_cast<int>(c.priority) < static_cast<int>(best->priority) ||
          (c.priority == best->priority && c.stamp > best->stamp))
        best = &c;
    }
    double target = best ? best->targets.at(dof.name) : dof.home;
    const Priority prio = best ? best->priority : Priority::Idle;
    std::optional<std::string> winner = best ? std::optional<std::string>(best->source) : std::nullopt;
    if (static_cast<int>(prio) >= static_cast<int>(Priority::Commanded)) {
      for (const auto& [_, c] : cmds_)
        if (c.additive && c.targets.count(dof.name)) target += c.targets.at(dof.name);
    }
    if (prio != Priority::Safety) {
      target = std::min(std::max(target, dof.min), dof.max);
      target = dof.home + (target - dof.home) * amplitude;
    }
    if (winner != winner_[dof.name]) {
      fade_[dof.name] = prio == Priority::Safety ? std::nullopt
                                                 : std::optional<Fade>(Fade{now, last_[dof.name]});
      winner_[dof.name] = winner;
    }
    auto& fade = fade_[dof.name];
    if (fade && crossfade_s_ > 0) {
      const double s = (now - fade->start) / crossfade_s_;
      if (s >= 1.0) {
        fade.reset();
      } else {
        target = fade->from_value + (target - fade->from_value) * curves::min_jerk(s);
      }
    }
    out[dof.name] = target;
  }
  last_ = out;
  return out;
}

// --------------------------------------------------------------------------- Limiter

Limiter::Limiter(std::vector<Dof> dofs, std::vector<YawDelta> constraints, double constraint_margin)
    : dofs_(std::move(dofs)), constraints_(std::move(constraints)), constraint_margin_(constraint_margin) {
  for (const auto& d : dofs_) {
    pos_[d.name] = d.home;
    vel_[d.name] = 0.0;
  }
}

void Limiter::reset(const Values& positions) {
  require_finite(positions);
  for (const auto& [name, p] : positions) {
    if (!pos_.count(name)) continue;
    pos_[name] = p;
    vel_[name] = 0.0;
  }
}

Values Limiter::clamp(const Values& targets) const {
  Values out;
  for (const auto& d : dofs_) {
    auto it = targets.find(d.name);
    if (it != targets.end()) out[d.name] = std::clamp(it->second, d.min, d.max);
  }
  for (const auto& c : constraints_) {
    if (out.count(c.a) && out.count(c.b))
      out[c.a] = std::clamp(out[c.a], out[c.b] - c.max, out[c.b] + c.max);
  }
  return out;
}

Values Limiter::step(const Values& targets, double dt, double factor) {
  if (!(std::isfinite(dt) && dt > 0)) throw std::invalid_argument("dt must be finite and > 0");
  if (!(std::isfinite(factor) && factor > 0 && factor <= 1))
    throw std::invalid_argument("factor must be in (0, 1]");
  require_finite(targets);
  Values goal = clamp(targets);
  // Coupled constraints act on goals (band around the other dof's current position, planned
  // constraint_margin inside the limit); no post-hoc position edits (as in Python).
  for (const auto& c : constraints_) {
    if (!pos_.count(c.a) || !pos_.count(c.b)) continue;
    const double lim = std::max(0.0, c.max - constraint_margin_);
    const std::pair<std::string, std::string> order[2] = {{c.a, c.b}, {c.b, c.a}};
    for (const auto& [u, w] : order) {
      auto g = goal.find(u);
      if (g == goal.end()) continue;
      const Dof* du = nullptr;
      for (const auto& d : dofs_) if (d.name == u) du = &d;
      const double v = std::clamp(g->second, pos_[w] - lim, pos_[w] + lim);
      g->second = std::clamp(v, du->min, du->max);
    }
  }
  const Values prev = pos_;
  // Velocity cap scaled by factor; braking always at full acceleration.
  for (const auto& d : dofs_) {
    auto g = goal.find(d.name);
    if (g == goal.end()) continue;
    const double vmax = d.max_vel * factor, amax = d.max_acc;
    const double x = pos_[d.name], v = vel_[d.name];
    const double err = g->second - x;
    const double v_cap = std::min(vmax, stop_speed(std::fabs(err), amax, dt));
    const double v_des = std::clamp(err / dt, -v_cap, v_cap);
    const double dv = std::clamp(v_des - v, -amax * dt, amax * dt);
    double x_new = x + (v + dv) * dt;
    if ((g->second - x_new) * err < 0 && std::fabs(err / dt - v) <= amax * dt * kLandTolerance)
      x_new = g->second;
    pos_[d.name] = x_new;
  }
  // Hard position limits; a dof starting outside is never pushed further out.
  for (const auto& d : dofs_) {
    const double x0 = prev.at(d.name);
    double& x = pos_[d.name];
    if (x0 >= d.min && x0 <= d.max) x = std::clamp(x, d.min, d.max);
    else if (x0 > d.max) x = std::min(x, x0);
    else x = std::max(x, x0);
  }
  for (const auto& d : dofs_) vel_[d.name] = (pos_[d.name] - prev.at(d.name)) / dt;
  return pos_;
}

// --------------------------------------------------------------------------- MotionCore

MotionCore::MotionCore(std::vector<Dof> dofs, std::vector<YawDelta> constraints, double crossfade_s,
                       double watchdog_s, double minor_factor, double watchdog_return_factor,
                       double constraint_margin)
    : arbiter_(dofs, crossfade_s),
      limiter_(dofs, std::move(constraints), constraint_margin),
      watchdog_s_(watchdog_s),
      minor_factor_(minor_factor),
      watchdog_return_factor_(watchdog_return_factor) {}

void MotionCore::feed(const Command& cmd) {
  arbiter_.submit(cmd);
  last_feed_ = cmd.stamp;
}

Values MotionCore::tick(double now, double dt) {
  if (last_feed_ && now - *last_feed_ > watchdog_s_) {
    if (!holding_) {
      Command home{"watchdog", Priority::Safety, {}, now, std::nullopt, false};
      for (const auto& d : arbiter_.dofs()) home.targets[d.name] = d.home;
      arbiter_.submit(home);
      holding_ = true;
    }
  } else if (holding_) {
    arbiter_.clear("watchdog");
    holding_ = false;
  }
  const double factor = minor_present_ ? minor_factor_ : 1.0;
  const Values targets = arbiter_.step(now, factor);
  const double speed = holding_ ? std::min(factor, watchdog_return_factor_) : factor;
  return limiter_.step(targets, dt, speed);
}

}  // namespace roboemot
