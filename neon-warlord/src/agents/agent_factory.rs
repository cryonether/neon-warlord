//! Instantiates agents

use crate::verlet_physics::verlet_composition::{LinkKind, Node};
use regex::Regex;
type Vec3 = cgmath::Vector3<f32>;

pub struct AgentFactory {
    re: Regex,
}

impl AgentFactory {
    pub fn new() -> Self {
        let re = Regex::new(r"(?P<id>\d+)(?:-(?P<kind>[A-Z])(?P<target>\d+))").unwrap();

        Self { re }
    }

    pub fn create_agent<const NR_SLICES: usize, const R: usize, const C: usize>(
        &self,
        layers: &[[[&'static str; C]; R]; NR_SLICES],
        pos: Vec3,
        scale: f32,
    ) -> Vec<Node> {
        let mut nodes = Vec::new();

        // Parse nodes
        #[allow(clippy::needless_range_loop)]
        for nr_slice in 0..NR_SLICES {
            for r in 0..R {
                for c in 0..C {
                    let content = layers[nr_slice][r][c];

                    if content.trim().is_empty() {
                        // elem is empty or contains only whitespace
                        continue;
                    }

                    let mut elem = self.parse(content);
                    elem.location = (nr_slice, r, c);

                    nodes.push(elem);
                }
            }
        }

        nodes.sort_by_key(|node| node.id);

        // Create result
        let mut res = Vec::with_capacity(nodes.len());
        if nodes.is_empty() {
            return res;
        }

        let origin = &nodes[0];
        let origin_pos = Vec3::new(
            origin.location.2 as f32,
            origin.location.1 as f32,
            origin.location.0 as f32,
        );

        for node in nodes {
            let local_pos = Vec3::new(
                node.location.2 as f32 - origin_pos.x,
                -(node.location.1 as f32 - origin_pos.y),
                node.location.0 as f32 - origin_pos.z,
            );

            res.push(Node {
                id: node.id,
                link_kind: node.link_kind,
                link_target: node.link_target,
                pos: local_pos,
            });
        }

        // move node
        for elem in &mut res {
            elem.pos = elem.pos * scale + pos;
        }

        res
    }

    fn parse(&self, elem: &str) -> AgentNode {
        let caps = self.re.captures(elem);

        let caps_ = match caps {
            Some(caps) => caps,
            None => {
                panic!("elem '{elem}' does not match");
            }
        };

        let id = &caps_["id"];
        let kind = &caps_["kind"];
        let target = &caps_["target"];

        let id: usize = id.parse().unwrap();
        let link_target: usize = target.parse().unwrap();

        let link_kind = match kind {
            "F" => LinkKind::Fixed,
            "L" => LinkKind::Linked,
            "S" => LinkKind::Sticky,
            "O" => LinkKind::Origin,
            &_ => panic!("Error parsing agent definition"),
        };

        AgentNode {
            id,
            link_kind,
            link_target,
            location: (0, 0, 0),
        }
    }
}

pub struct AgentNode {
    pub id: usize,
    pub link_kind: LinkKind,
    pub link_target: usize,
    pub location: (usize, usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_agent_definition() {
        let factory = AgentFactory::new();

        let elem = "13-S9";

        let caps = factory.re.captures(elem).expect("regex should match");

        assert_eq!(&caps["id"], "13");
        assert_eq!(&caps["kind"], "S");
        assert_eq!(&caps["target"], "9");
    }
}
