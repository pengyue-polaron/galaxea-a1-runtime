// Private stdin/stdout adapter to the installed upstream TRAC-IK library.
// No ROS node, command publisher, device access, or trajectory generation.
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <trac_ik/trac_ik.hpp>
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <set>
#include <limits>

using Json = nlohmann::json;

KDL::JntArray vector(const Json& data, size_t count) {
  if (!data.is_array() || data.size() != count)
    throw std::runtime_error("Invalid joint vector length");
  KDL::JntArray result(count);
  for (size_t i = 0; i < count; ++i) {
    result(i) = data.at(i).get<double>();
    if (!std::isfinite(result(i))) throw std::runtime_error("Non-finite vector");
  }
  return result;
}

int main() {
  try {
    std::string line;
    if (!std::getline(std::cin, line)) return 0;
    const auto init = Json::parse(line);
    KDL::Tree tree;
    KDL::Chain chain;
    if (!kdl_parser::treeFromString(init.at("urdf"), tree) ||
        !tree.getChain(init.at("base_link"), init.at("tip_link"), chain))
      throw std::runtime_error("Cannot construct the requested URDF chain");
    std::vector<std::string> names;
    for (const auto& segment : chain.segments)
      if (segment.getJoint().getType() != KDL::Joint::None)
        names.push_back(segment.getJoint().getName());
    if (names != init.at("joint_names").get<std::vector<std::string>>() ||
        std::set<std::string>(names.begin(), names.end()).size() != names.size())
      throw std::runtime_error("URDF chain joint order differs from System");
    const size_t count = names.size();
    auto lower = vector(init.at("lower"), count);
    auto upper = vector(init.at("upper"), count);
    const double timeout = init.at("timeout_s"), epsilon = init.at("epsilon");
    if (!(std::isfinite(timeout) && timeout > 0 && std::isfinite(epsilon) && epsilon > 0))
      throw std::runtime_error("Invalid solver precision or timeout");
    for (size_t i = 0; i < count; ++i)
      if (lower(i) >= upper(i)) throw std::runtime_error("Invalid joint limits");
    TRAC_IK::TRAC_IK solver(chain, lower, upper, timeout, epsilon, TRAC_IK::Distance);
    const auto tolerances = vector(init.at("bounds"), 6);
    const double position_tolerance = init.at("position_tolerance");
    const double orientation_tolerance = init.at("orientation_tolerance");
    KDL::Twist bounds;
    for (size_t i = 0; i < 6; ++i) {
      if (tolerances(i) < 0) throw std::runtime_error("Invalid Cartesian bounds");
      bounds(i) = tolerances(i);
    }
    KDL::ChainFkSolverPos_recursive fk(chain);
    std::cout << Json({{"ready", true}, {"joint_names", names}}).dump() << std::endl;
    while (std::getline(std::cin, line)) {
      const auto request = Json::parse(line);
      const auto seed = vector(request.at("seed"), count);
      const double delta = request.at("delta_limit");
      if (!std::isfinite(delta) || delta <= 0) throw std::runtime_error("Invalid delta");
      KDL::JntArray lo(count), hi(count), result(count);
      for (size_t i = 0; i < count; ++i) {
        if (seed(i) < lower(i) || seed(i) > upper(i))
          throw std::runtime_error("Seed outside absolute joint limits");
        lo(i) = std::max(lower(i), seed(i) - delta);
        hi(i) = std::min(upper(i), seed(i) + delta);
      }
      solver.setKDLLimits(lo, hi);
      const auto pose = vector(request.at("pose"), 7);
      double norm = 0;
      for (size_t i = 3; i < 7; ++i) norm += pose(i) * pose(i);
      norm = std::sqrt(norm);
      if (norm <= 0) throw std::runtime_error("Zero quaternion");
      const KDL::Frame goal(KDL::Rotation::Quaternion(
          pose(3)/norm, pose(4)/norm, pose(5)/norm, pose(6)/norm),
          KDL::Vector(pose(0), pose(1), pose(2)));
      int code = solver.CartToJnt(seed, goal, result, bounds);
      if (code >= 0) {
        // TRAC-IK exposes axis bounds; Runtime's acceptance uses vector norms.
        // Choose the nearest native candidate that passes those same norm gates.
        std::vector<KDL::JntArray> candidates;
        solver.getSolutions(candidates);
        double best = std::numeric_limits<double>::infinity();
        code = -3;
        for (const auto& candidate : candidates) {
          KDL::Frame frame;
          if (fk.JntToCart(candidate, frame) < 0) throw std::runtime_error("FK failure");
          const auto error = KDL::diff(frame, goal);
          if (error.vel.Norm() > position_tolerance || error.rot.Norm() > orientation_tolerance)
            continue;
          bool within_limits = true;
          for (size_t i = 0; i < count; ++i)
            if (candidate(i) < lo(i) || candidate(i) > hi(i)) within_limits = false;
          const double cost = TRAC_IK::TRAC_IK::JointErr(seed, candidate);
          if (within_limits && std::isfinite(cost) && cost < best) {
            best = cost;
            result = candidate;
            code = 0;
          }
        }
      }
      Json reply = {{"id", request.at("id")}, {"code", code}};
      if (code >= 0) {
        reply["positions"] = std::vector<double>(result.data.data(), result.data.data() + count);
        KDL::Frame frame;
        if (fk.JntToCart(result, frame) < 0) throw std::runtime_error("FK failure");
        double x, y, z, w;
        frame.M.GetQuaternion(x, y, z, w);
        reply["fk"] = {frame.p.x(), frame.p.y(), frame.p.z(), x, y, z, w};
      }
      std::cout << reply.dump() << std::endl;
    }
  } catch (const std::exception& error) {
    std::cerr << "A1 TRAC-IK worker: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
