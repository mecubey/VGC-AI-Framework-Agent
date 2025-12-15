import numpy as np
from vgc2.battle_engine.game_state import Side, State
from vgc2.battle_engine.modifiers import Weather, Terrain, Hazard, Stat, Status
from vgc2.battle_engine.move import Move, BattlingMove
from vgc2.battle_engine.pokemon import BattlingPokemon
from vgc2.battle_engine.damage_calculator import calculate_damage
from vgc2.battle_engine.constants import BattleRuleParam
from vgc2.util.encoding import EncodeContext, one_hot
from vgc2.agent import BattlePolicy

class ENCODING_CONSTANTS:
    MAX_TEAMS = 2
    MAX_ACT_PKM_PER_TEAM = 2
    MAX_PKM_PER_TEAM = 4
    MAX_MOVES_PER_PKM = 4
    MOVE = 37
    EMBEDDED_MOVE = 16
    STATE = 439

def encode_move(e: np.array, move: Move, ctx: EncodeContext) -> int:
    # for enemy moves:
    # encode target and dmg done to target

    # for own moves:
    # encode dmg done to both enemies (0 if enemy is dead)

    i = 0
    e[i] = move.accuracy
    i += 1
    e[i] = move.priority / ctx.max_priority
    i += 1
    e[i] = move.effect_prob
    i += 1
    e[i] = float(move.force_switch)
    i += 1
    e[i] = float(move.self_switch)
    i += 1
    e[i] = float(move.ignore_evasion)
    i += 1
    e[i] = float(move.protect)
    i += 1
    e[i] = move.heal / ctx.max_ratio
    i += 1
    e[i] = move.recoil / ctx.max_ratio
    i += 1
    e[i] = float(move.toggle_trickroom)
    i += 1
    e[i] = float(move.toggle_reflect)
    i += 1
    e[i] = float(move.toggle_lightscreen)
    i += 1
    e[i] = float(move.toggle_tailwind)
    i += 1
    e[i] = float(move.change_type)
    i += 1
    e[i] = float(move.disable)
    i += 1
    e[i] = move.boosts[Stat.SPEED] / ctx.max_boost
    i += 1
    e[i] = move.boosts[Stat.EVASION] / ctx.max_boost
    i += 1
    e[i] = move.boosts[Stat.ACCURACY] / ctx.max_boost
    i += 1
    if move.weather_start != Weather.CLEAR:
        one_hot(e[i:], move.weather_start - 1, ctx.n_weather)
    i += ctx.n_weather
    if move.field_start != Terrain.NONE:
        one_hot(e[i:], move.field_start - 1, ctx.n_terrain)
    i += ctx.n_terrain
    if move.hazard != Hazard.NONE:
        one_hot(e[i:], move.hazard - 1, ctx.n_hazard)
    i += ctx.n_hazard
    if move.status != Status.NONE:
        one_hot(e[i:], move.status - 1, ctx.n_status)
    i += ctx.n_status
    return i

def encode_battling_move(e: np.array, move: BattlingMove, ctx: EncodeContext) -> int:
    i = encode_move(e, move.constants, ctx)
    e[i] = float(move.disabled)
    i += 1
    e[i] = move.pp / ctx.max_pp
    i += 1
    return i

def encode_battling_pokemon(e: np.array, pokemon: BattlingPokemon, ctx: EncodeContext) -> int:
    i = 0

    # Remaining HP
    e[i] = pokemon.hp / pokemon.constants.stats[Stat.MAX_HP]
    i += 1

    # Important stats
    e[i] = pokemon.constants.stats[Stat.SPEED] / ctx.max_hp
    i += 1

    # Important boosts
    e[i] = pokemon.boosts[Stat.SPEED]
    i += 1
    e[i] = pokemon.boosts[Stat.EVASION]
    i += 1
    e[i] = pokemon.boosts[Stat.ACCURACY]
    i += 1

    # Protect
    e[i] = float(pokemon.protect)
    i += 1

    # wake turns
    e[i] = pokemon._wake_turns / ctx.max_sleep
    i += 1

    if pokemon.status != Status.NONE:
        one_hot(e[i:], pokemon.status, ctx.n_status)
    i += ctx.n_status
    return i

def encode_battling_team(e: np.array, team: list[BattlingPokemon], ctx: EncodeContext) -> int:
    i = 0
    pkm_enc_len = 0 
    for m in team:
        pkm_enc_len = encode_battling_pokemon(e[i:], m, ctx)
        i += pkm_enc_len
    if len(team)==1:
        i += pkm_enc_len # We want to keep a consistent encoding length, so if there is only one pokemon left on the team, 
                         # features of the second non-existing pokemon will all be encoded as 0
    return i

def encode_side(e: np.array, side: Side, ctx: EncodeContext) -> int:
    i = 0
    i += encode_battling_team(e[i:], side.team.active, ctx)
    e[i] = float(side.conditions.reflect)
    i += 1
    e[i] = float(side.conditions.lightscreen)
    i += 1
    e[i] = float(side.conditions.tailwind)
    i += 1
    e[i] = float(side.conditions.stealth_rock)
    i += 1
    e[i] = float(side.conditions.poison_spikes)
    i += 1
    return i

def encode_state(e: np.array, state: State, ctx: EncodeContext, params: BattleRuleParam, opponent: BattlePolicy) -> int:
    i = 0
    own_team = state.sides[0].team.active+state.sides[0].team.reserve
    opp_team = state.sides[1].team.active+state.sides[1].team.reserve

    # Sides
    for s in state.sides:
        i += encode_side(e[i:], s, ctx) # This doesn't encode pokemon moves

    if state.weather != Weather.CLEAR:
        one_hot(e[i:], state.weather - 1, ctx.n_weather)
    i += ctx.n_weather
    if state.field != Terrain.NONE:
        one_hot(e[i:], state.field - 1, ctx.n_terrain)
    i += ctx.n_terrain
    e[i] = float(state.trickroom)
    i += 1

    # hp calc
    e[i] = sum([pkm.hp for pkm in own_team]) / sum([pkm.constants.stats[Stat.MAX_HP] for pkm in own_team])
    i += 1
    e[i] = sum([pkm.hp for pkm in opp_team]) / sum([pkm.constants.stats[Stat.MAX_HP] for pkm in opp_team])
    i += 1

    # Encode my pkm's moves
    for battle_pkm in state.sides[0].team.active:
        for m in battle_pkm.battling_moves:
            for defender in state.sides[1].team.active:
                e[i] = calculate_damage(params, 0, m.constants, state, battle_pkm, defender) / ctx.max_hp
                i += 1
            if len(state.sides[1].team.active)==1:
                i += 1 # Keep state observation consistent length
            i += encode_battling_move(e[i:], m, ctx)
    if len(state.sides[0].team.active) == 1:
        i += ENCODING_CONSTANTS.MOVE*ENCODING_CONSTANTS.MAX_MOVES_PER_PKM # Keep state observation consistent length
    
    # encode predicted (greedy) action of opponent
    opp_state = State(tuple((state.sides[1], state.sides[0])))
    act_opp_pkm = opp_state.sides[0].team.active
    have_seen_moves = all(len(p.battling_moves) > 0 for p in act_opp_pkm)
    if have_seen_moves:
        opp_decision = opponent.decision(opp_state)
        for j, action in enumerate(opp_decision):
            move = act_opp_pkm[j].battling_moves[action[0]]
            target = opp_state.sides[1].team.active[action[1]]
            e[i] = calculate_damage(params, 0, move.constants, opp_state, act_opp_pkm[j], target) / ctx.max_hp
            i += 1
            e[i] = action[1]
            i += 1
            i += encode_battling_move(e[i:], move, ctx)
        if len(opp_decision) == 1:
            i += ENCODING_CONSTANTS.MOVE
    else:
        i += 2 * ENCODING_CONSTANTS.MOVE

    return e