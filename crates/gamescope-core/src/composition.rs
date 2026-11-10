//! Multi-plane Wayland / Game + Overlay layer composition plan.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::scaling::ViewportRect;

/// Composition blending modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BlendMode {
    Opaque,
    PremultipliedAlpha,
    StraightAlpha,
    Additive,
}

/// Overlay / Plane classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LayerType {
    BaseGame,
    SteamOverlay,
    MangoHud,
    WaylandSubsurface,
    Custom,
}

/// Representation of an active micro-compositor layer plane.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompositorLayer {
    pub id: u32,
    pub name: String,
    pub layer_type: LayerType,
    pub z_index: i32,
    pub visible: bool,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub source_rect: ViewportRect,
    pub dest_rect: ViewportRect,
}

impl CompositorLayer {
    pub fn new_base_game(id: u32, width: f32, height: f32) -> Self {
        Self {
            id,
            name: "BaseGame".into(),
            layer_type: LayerType::BaseGame,
            z_index: 0,
            visible: true,
            opacity: 1.0,
            blend_mode: BlendMode::Opaque,
            source_rect: ViewportRect::new(0.0, 0.0, width, height),
            dest_rect: ViewportRect::new(0.0, 0.0, width, height),
        }
    }

    pub fn new_steam_overlay(id: u32, width: f32, height: f32) -> Self {
        Self {
            id,
            name: "SteamOverlay".into(),
            layer_type: LayerType::SteamOverlay,
            z_index: 100,
            visible: true,
            opacity: 1.0,
            blend_mode: BlendMode::PremultipliedAlpha,
            source_rect: ViewportRect::new(0.0, 0.0, width, height),
            dest_rect: ViewportRect::new(0.0, 0.0, width, height),
        }
    }

    pub fn new_mangohud(id: u32, width: f32, height: f32) -> Self {
        Self {
            id,
            name: "MangoHud".into(),
            layer_type: LayerType::MangoHud,
            z_index: 200,
            visible: true,
            opacity: 0.95,
            blend_mode: BlendMode::PremultipliedAlpha,
            source_rect: ViewportRect::new(0.0, 0.0, width, height),
            dest_rect: ViewportRect::new(16.0, 16.0, width, height),
        }
    }
}

/// Composition Plan generated for the micro-compositor frame renderer.
#[derive(Debug, Clone, Default)]
pub struct CompositionPlan {
    layers: Vec<CompositorLayer>,
}

impl CompositionPlan {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: CompositorLayer) {
        self.layers.push(layer);
        self.sort_by_z();
    }

    pub fn sort_by_z(&mut self) {
        self.layers.sort_by_key(|l| l.z_index);
    }

    pub fn active_layers(&self) -> impl Iterator<Item = &CompositorLayer> {
        self.layers.iter().filter(|l| l.visible && l.opacity > 0.0)
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn clear(&mut self) {
        self.layers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition_plan_z_ordering() {
        let mut plan = CompositionPlan::new();
        let hud = CompositorLayer::new_mangohud(2, 300.0, 200.0);
        let game = CompositorLayer::new_base_game(1, 1920.0, 1080.0);
        let steam = CompositorLayer::new_steam_overlay(3, 1920.0, 1080.0);

        plan.add_layer(hud);
        plan.add_layer(steam);
        plan.add_layer(game);

        let active: Vec<&CompositorLayer> = plan.active_layers().collect();
        assert_eq!(active.len(), 3);
        assert_eq!(active[0].layer_type, LayerType::BaseGame);
        assert_eq!(active[1].layer_type, LayerType::SteamOverlay);
        assert_eq!(active[2].layer_type, LayerType::MangoHud);
    }
}
