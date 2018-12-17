import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

'''Read in SQL-tables'''

database = '../data/database.sqlite'
conn = sqlite3.connect(database)

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)

countries = pd.read_sql("""SELECT *
                        FROM Country;""", conn)

leagues = pd.read_sql("""SELECT Country.id, League.name AS "League name", \
                        Country.name AS "Country name"
                        FROM League
                        JOIN Country ON Country.id = League.country_id;""", conn)

team_stats = pd.read_sql("""SELECT Team.team_api_id, Team.team_fifa_api_id, \
                    team_long_name, team_short_name, \
                    Team_attributes.id AS "team_attributes.id", \
                    date, buildUpPlaySpeed, buildUpPlaySpeedClass, \
                    buildUpPlayDribbling, buildUpPlayDribblingClass, \
                    buildUpPlayPassing, buildUpPlayPassingClass, \
                    buildUpPlayPositioningClass, chanceCreationPassing, \
                    chanceCreationPassingClass, chanceCreationCrossing, \
                    chanceCreationCrossingClass, chanceCreationShooting, \
                    chanceCreationShootingClass, chanceCreationPositioningClass, \
                    defencePressure, defencePressureClass, defenceAggression, \
                    defenceAggressionClass, defenceTeamWidth, defenceTeamWidthClass, \
                    defenceDefenderLineClass
                    FROM Team
                    JOIN Team_attributes
                    ON Team.team_api_id = Team_attributes.team_api_id;""", conn)

player_stats = pd.read_sql("""SELECT Player.player_api_id,  Player.player_fifa_api_id, \
                      player_name, birthday, height, weight, date, overall_rating, \
                      potential, preferred_foot, attacking_work_rate, defensive_work_rate, \
                      crossing, finishing, heading_accuracy, short_passing, \
                      volleys, dribbling, curve, free_kick_accuracy, long_passing, \
                      ball_control, acceleration, sprint_speed, agility, reactions, \
                      balance, shot_power, jumping, stamina, strength, long_shots, \
                      aggression, interceptions, positioning, vision, penalties, \
                      marking, standing_tackle, sliding_tackle, gk_diving, gk_handling, \
                      gk_kicking, gk_positioning, gk_reflexes
                      FROM Player
                      JOIN Player_attributes
                      ON Player.player_api_id = Player_attributes.player_api_id
                      ;""", conn)

matches = pd.read_sql("""WITH leagues AS

                      (
                      SELECT Country.id, League.name AS "league_name", \
                      Country.name AS "country_name"
                      FROM League
                      JOIN Country ON Country.id = League.country_id
                      )

                      SELECT Match.id, league_name, season, stage, date, \
                      match_api_id, home_team_api_id, away_team_api_id, \
                      home_team_goal, away_team_goal, home_player_1, home_player_2, home_player_3, \
                      home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, \
                      home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, \
                      away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, \
                      away_player_8, away_player_9, away_player_10, away_player_11, goal, shoton, \
                      shotoff, foulcommit, card, cross, corner, possession
                      FROM Match
                      JOIN leagues ON Match.league_id = leagues.id
                      ;""", conn)

''' Functions for feature extraction '''


''' Derives a label for a given match. '''
def get_match_label(match):

    #Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    label = pd.DataFrame()
    label.loc[0,'match_api_id'] = match['match_api_id']

    #Identify match label
    if home_goals > away_goals:
        label.loc[0,'label'] = "Win"
    if home_goals == away_goals:
        label.loc[0,'label'] = "Draw"
    if home_goals < away_goals:
        label.loc[0,'label'] = "Defeat"

    #Return label
    return label.loc[0]


'''Get all available player stats for each player'''
def get_player_stats_extended(match, player_stats):

    #Define variables
    match_id =  match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    player_var_names = []

    #Get dummy variables

    dummies = pd.get_dummies(player_stats['preferred_foot'])
    match_stats = pd.concat([player_stats, dummies], axis = 1)
    match_stats.drop(['preferred_foot'], inplace = True, axis = 1)

    dummies = pd.get_dummies(player_stats['attacking_work_rate'])
    match_stats = pd.concat([player_stats, dummies], axis = 1)
    match_stats.drop(['attacking_work_rate'], inplace = True, axis = 1)

    dummies = pd.get_dummies(player_stats['defensive_work_rate'])
    match_stats = pd.concat([player_stats, dummies], axis = 1)
    match_stats.drop(['defensive_work_rate'], inplace = True, axis = 1)

    #Loop through all players
    for player in players:

        #Get player ID
        player_id = match[player]

        #Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]

        #Identify current stats
        current_stats = stats[stats.date < date].sort_values(by = 'date', ascending = False)[:1]

        current_stats.drop(['player_fifa_api_id'], inplace = True, axis = 1)
        current_stats.drop(['player_name'], inplace = True, axis = 1)
        current_stats.drop(['date'], inplace = True, axis = 1)
        current_stats.loc[:, 'birthday'] = pd.to_datetime(current_stats.birthday)

        stat_variables = list(current_stats)

        for stat in stat_variables:

            if np.isnan(player_id) == True:
                value = pd.Series(0)
            else:
                current_stats.reset_index(inplace = True, drop = True)
                value = pd.Series(current_stats.loc[0, stat])

            #Rename stat
            name = "{}_{}".format(player, stat)
            player_var_names.append(name)

            #Aggregate stats
            player_stats_new = pd.concat([player_stats_new, value], axis = 1)

    player_stats_new.columns = player_var_names
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace = True, drop = True)

    #Return player stats
    return player_stats_new.ix[0]

'''
Find overall rating for each player
'''

def get_player_stats(match, player_stats):

    match_id =  match.match_api_id
    date = match['date']

    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    player_var_names = []

    for player in players:

        player_id = match[player]
        stats = player_stats[player_stats.player_api_id == player_id]
        current_stats = stats[stats.date < date].sort_values(by = 'date', ascending = False)[:1]

        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace = True, drop = True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])

        name = "{}_overall_rating".format(player)
        player_var_names.append(name)

        player_stats_new = pd.concat([player_stats_new, overall_rating], axis = 1)

    player_stats_new.columns = player_var_names
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace = True, drop = True)

    #Return player stats
    return player_stats_new.ix[0]


''' Get the last x matches of a given team. '''
def get_last_matches(matches, date, team, x = 10):

    #Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]

    #Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]

    #Return last matches
    return last_matches

''' Get the last x matches between two given teams. '''
def get_last_matches_against_eachother(matches, date, home_team, away_team, x = 10):

    #Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]
    total_matches = pd.concat([home_matches, away_matches])

    #Get last x matches
    try:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:total_matches.shape[0],:]

        #Check for error in data
        if(last_matches.shape[0] > x):
            print("Error in obtaining matches")

    #Return data
    return last_matches

''' Get the goals conceded of a specfic team from a set of matches. '''
def get_goals_conceded(matches, team):

    #Find home and away goals
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    #Return total goals
    return total_goals

''' Get the goals of a specfic team from a set of matches. '''
def get_goals(matches, team):

    #Find home and away goals
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    #Return total goals
    return total_goals

''' Get the number of wins of a specfic team from a set of matches. '''
def get_wins(matches, team):

    #Find home and away wins
    home_wins = int(matches.home_team_goal[(matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[(matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())

    total_wins = home_wins + away_wins

    #Return total wins
    return total_wins

''' Create match specific features for a given match. '''
def get_match_features(match, matches, x = 10):

    #Define variables
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id

    #Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, date, home_team, x = 10)
    matches_away_team = get_last_matches(matches, date, away_team, x = 10)

    #Get last x matches of both teams against each other
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x = 10)

    #Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceded = get_goals_conceded(matches_home_team, home_team)
    away_goals_conceded = get_goals_conceded(matches_away_team, away_team)

    #Define result data frame
    result = pd.DataFrame()

    #Define mathch features
    result.loc[0, 'match_api_id'] = match.match_api_id
    result.loc[0, 'league_name'] = match.league_name
    result.loc[0, 'season'] = match.season
    result.loc[0, 'stage'] = match.stage

    #Create new match features
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceded
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceded
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team)
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)

    #Return match features
    return result.loc[0]

'''Creates and aggregates the features'''
def create_features(matches, fifa_stats, horizontal = True, x = 10):

    match_stats = matches.apply(lambda x: get_match_features(x, matches, x = 10), axis = 1)

    #Create match labels
    labels = matches.apply(get_match_label, axis = 1)

    dummies = pd.get_dummies(match_stats['league_name'])
    match_stats = pd.concat([match_stats, dummies], axis = 1)
    match_stats.drop(['league_name'], inplace = True, axis = 1)

    dummies = pd.get_dummies(match_stats['season'])
    match_stats = pd.concat([match_stats, dummies], axis = 1)
    match_stats.drop(['season'], inplace = True, axis = 1)

    #Merges features and labels into one frame
    features = pd.merge(match_stats, fifa_stats, on = 'match_api_id', how = 'left')
    feables = pd.merge(features, labels, on = 'match_api_id', how = 'left')

    #Drop NA values
    feables.dropna(inplace = True)

    #Return preprocessed data
    return feables


#start_time = time.time()

'''NB: This takes a very long time..'''
fifa_data = matches.apply(lambda x : get_player_stats(x, player_stats), axis = 1)
features_labels = create_features(matches, fifa_data)
final_data = features_labels.drop('match_api_id', axis = 1)

#Save data
#final_data.to_pickle('../data/preprocessed_data.pkl')

#print("--- %s seconds ---" % (time.time() - start_time))
