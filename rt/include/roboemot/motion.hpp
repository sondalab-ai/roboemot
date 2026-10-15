// roboemot-rt: arbiter, safety limiter and motion core (SPEC 10, 12).
// Same semantics as packages/roboemot-core/src/roboemot/motion/core.py, checked against
// golden vectors generated from the Python reference (scripts/gen_golden.py).
#pragma once

#include <map>
#include <utility>
#include <optional>
#include <string>
#include <vector>

namespace roboemot {

enum class Priority : int { Safety = 0, Reflex = 1, Commanded = 2, Mimicry = 3, Expression = 4, Idle = 5 };

struct Dof {
  std::string name;
  double min = 0, max = 0, max_vel = 0, max_acc = 0, home = 0;
};

struct YawDelta {  // |a - b| <= max (rad); a is adjusted
  std::string a, b;
  double max = 0;
};

struct Command {
  std::string source;
  Priority priority = Priority::Idle;
  std::map<std::string, double> targets;
  double stamp = 0;
  std::optional<double> expires;
  bool additive = false;
};

using Values = std::map<std::string, double>;

namespace curves {
double linear(double s);
double min_jerk(double s);
double ease_in_out(double s);
double cartoon(double s);
}  // namespace curves

double stop_distance(double v, double adt, double dt);
double stop_speed(double distance, double amax, double dt);

class Arbiter {
 public:
  Arbiter(std::vector<Dof> dofs, double crossfade_s);
  void submit(const Command& cmd);  // throws std::out_of_range on unknown dofs
  void clear(const std::string& source);
  Values step(double now, double amplitude = 1.0);
  const std::vector<Dof>& dofs() const { return dofs_; }

 private:
  struct Fade {
    double start, from_value;
  };
  std::vector<Dof> dofs_;
  double crossfade_s_;
  std::map<std::string, Command> cmds_;
  std::map<std::string, std::optional<std::string>> winner_;
  std::map<std::string, std::optional<Fade>> fade_;
  Values last_;
};

class Limiter {
 public:
  Limiter(std::vector<Dof> dofs, std::vector<YawDelta> constraints, double constraint_margin = 0.0);
  Values clamp(const Values& targets) const;
  // Throws std::invalid_argument on non-finite targets, dt <= 0 or factor outside (0, 1].
  Values step(const Values& targets, double dt, double factor = 1.0);
  // Re-align to measured positions (start-up, return from teach/limp): velocity becomes zero.
  void reset(const Values& positions);
  const Values& position() const { return pos_; }

 private:
  std::vector<Dof> dofs_;
  std::vector<YawDelta> constraints_;
  double constraint_margin_ = 0.0;  // rad; coupled constraints planned this far inside
  Values pos_, vel_;
};

class MotionCore {
 public:
  MotionCore(std::vector<Dof> dofs, std::vector<YawDelta> constraints, double crossfade_s,
             double watchdog_s, double minor_factor, double watchdog_return_factor,
             double constraint_margin = 0.0);
  void feed(const Command& cmd);
  Values tick(double now, double dt);
  void set_minor_present(bool v) { minor_present_ = v; }
  bool holding() const { return holding_; }

 private:
  Arbiter arbiter_;
  Limiter limiter_;
  double watchdog_s_, minor_factor_, watchdog_return_factor_;
  bool minor_present_ = false, holding_ = false;
  std::optional<double> last_feed_;
};

}  // namespace roboemot
