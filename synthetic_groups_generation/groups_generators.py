from abc import ABC, abstractmethod

import random
import numpy as np
import settings.config as cfg

random.seed(1010101)

# generation of random, similar and divergent groups revised from Kaya et al
# https://github.com/mesutkaya/recsys2020/blob/8a8c7088bebc3309b8517f62248386ea7be39776/GFAR_python/create_group.py

class GroupsGenerator(ABC):

    @staticmethod
    def getGroupsGenerator(type):
        if type == "RANDOM":
            return RandomGroupsGenerator()
        elif type == "SIMILAR":
            return SimilarGroupsGenerator()
        elif type == "DIVERGENT":
            return DivergentGroupsGenerator()
        elif type == "SIMILAR_ONE_DIVERGENT":
            return MinorityGroupsGenerator()
        elif type == "COUNTER_EXAMPLE":
            return CounterExampleGroupsGenerator()
        elif type == "BALANCED_COUNTER_EXAMPLE":
            return BalancedCounterExampleGroupsGenerator() # uses different bounds (mean, max)
        elif type == "INCREASING_SIMILARITY":
            return IncreasingSimilarityGroupsGenerator() # uses different bounds (min/2, min)
        elif type == "TWO_SUBGROUPS":
            return TwoSubGroupsGenerator() # generate two similar subgroups
        elif type == "OUTLIERS":
            return Outliers() # TODO also add DifferentOutliers where each outlier is as different from all the group members as possible
        return None

    @staticmethod
    def compute_average_similarity(group, user_id_indexes, sim_matrix):
        similarities = list()
        for user_1 in group:
            user_1_index = user_id_indexes.tolist().index(user_1)
            for user_2 in group:
                user_2_index = user_id_indexes.tolist().index(user_2)
                if user_1 != user_2:
                    similarities.append(sim_matrix[user_1_index][user_2_index])
        return np.mean(similarities)

    @abstractmethod
    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        pass


class RandomGroupsGenerator(GroupsGenerator):

    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        groups_list = list()
        for group_size in group_sizes_to_create:
            for i in range(group_number_to_create):
                group = random.sample(user_id_set, group_size)
                groups_list.append(
                    {
                        "group_size": group_size,
                        "group_similarity": 'random',
                        "group_members": group,
                        "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                    }
                )
            print(len(groups_list))
        return groups_list


class SimilarGroupsGenerator(GroupsGenerator):

    @staticmethod
    def select_user_for_sim_group(group, sim_matrix, user_id_indexes, sim_threshold=0.4):
        '''
        Helper function to the generate_similar_user_groups function. Given already selected group members, it randomly
        selects from the remaining users that has a PCC value >= sim_threshold to any of the existing members.
        :param group:
        :param sim_matrix:
        :param user_id_indexes:
        :param sim_threshold:
        :return:
        '''
        #print(set(user_id_indexes))
        ids_to_select_from = set(user_id_indexes) # in original version, this was: set()
        print(type(user_id_indexes))
        
        user_id_indexes_list = user_id_indexes.tolist()

        group_members = np.array([user_id_indexes_list.index(member) for member in group])

        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size
        print(np.sort(mean_similarities)[::-1][:10])
        print(sim_threshold)
        indexes = np.where(mean_similarities >= sim_threshold)[0].tolist()
        ids_to_select_from = [user_id_indexes[index] for index in indexes]

        # for member in group:
        #     member_index = user_id_indexes.tolist().index(member)
        #     indexes = np.where(sim_matrix[member_index] >= sim_threshold)[0].tolist()
        #     user_ids = [user_id_indexes[index] for index in indexes]
        #     ids_to_select_from = ids_to_select_from.union(set(user_ids)) # .union in original implementation
        candidate_ids = set(ids_to_select_from).difference(group)
        if len(candidate_ids) == 0:
            return None
        else:
            selection = random.sample(candidate_ids, 1)
            return selection[0]
        
    
    @staticmethod
    def select_users_for_sim_group(group, sim_matrix, user_id_indexes, n_to_select, sim_threshold=0.4):
        '''
        Helper function to the generate_similar_user_groups function. Given already selected group members, it randomly
        selects from the remaining users that has a PCC value >= sim_threshold to any of the existing members.
        :param group:
        :param sim_matrix:
        :param user_id_indexes:
        :param sim_threshold:
        :return:
        '''
        #print(set(user_id_indexes))
        ids_to_select_from = set(user_id_indexes) # in original version, this was: set()
        user_id_indexes_list = user_id_indexes.tolist()

        group_members = np.array([user_id_indexes_list.index(member) for member in group])

        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size
        indexes = np.where(mean_similarities >= sim_threshold)[0].tolist()
        ids_to_select_from = [user_id_indexes[index] for index in indexes]

        # for member in group:
        #     member_index = user_id_indexes.tolist().index(member)
        #     indexes = np.where(sim_matrix[member_index] >= sim_threshold)[0].tolist()
        #     user_ids = [user_id_indexes[index] for index in indexes]
        #     ids_to_select_from = ids_to_select_from.union(set(user_ids)) # .union in original implementation
        candidate_ids = set(ids_to_select_from).difference(group)
        if len(candidate_ids) < n_to_select:
            return None
        else:
            selection = random.sample(candidate_ids, n_to_select)
            return selection

    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        groups_list = list()
        for group_size in group_sizes_to_create:
            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                while len(group) < group_size:
                    # new_member = SimilarGroupsGenerator.select_user_for_sim_group(group, similarity_matrix,
                    #                                                               user_id_indexes,
                    #                                                               sim_threshold=cfg.similar_threshold)
                    # if new_member is None:
                    #     break
                    # group.append(new_member)
                    new_member = SimilarGroupsGenerator.select_users_for_sim_group(group, similarity_matrix,
                                                                                  user_id_indexes, n_to_select=group_size,
                                                                                  sim_threshold=cfg.similar_threshold)
                    if new_member:
                        group = new_member
                    else:
                        break                    
                if len(group) == group_size:
                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'similar',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )
            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list


class DivergentGroupsGenerator(GroupsGenerator):

    @staticmethod
    def select_user_for_divergent_group(group, sim_matrix, user_id_indexes, sim_threshold=0.0):
        '''
        Helper function to the generate_similar_user_groups function. Given already selected group members, it randomly
        selects from the remaining users that has a PCC value < sim_threshold to any of the existing members.
        :param group:
        :param sim_matrix:
        :param user_id_indexes:
        :param sim_threshold:
        :return:
        '''
        ids_to_select_from = set()
        for member in group:
            member_index = user_id_indexes.tolist().index(member)
            indexes = np.where(sim_matrix[member_index] < sim_threshold)[0].tolist()
            user_ids = [user_id_indexes[index] for index in indexes]
            ids_to_select_from = ids_to_select_from.union(set(user_ids))
        candidate_ids = ids_to_select_from.difference(set(group))
        if len(candidate_ids) == 0:
            return None
        else:
            selection = random.sample(candidate_ids, 1)
            return selection[0]

    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        groups_list = list()
        for group_size in group_sizes_to_create:
            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                while len(group) < group_size:
                    new_member = DivergentGroupsGenerator.select_user_for_divergent_group(group, similarity_matrix,
                                                                                     user_id_indexes,
                                                                                     sim_threshold=cfg.dissimilar_threshold)
                    if new_member is None:
                        break
                    group.append(new_member)
                if len(group) == group_size:
                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'divergent',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )
            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list


class MinorityGroupsGenerator(GroupsGenerator):
    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        groups_list = list()
        for group_size in group_sizes_to_create:
            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                while len(group) < (group_size - 1):
                    new_member = SimilarGroupsGenerator.select_user_for_sim_group(group, similarity_matrix,
                                                                                     user_id_indexes,
                                                                                     sim_threshold=cfg.similar_threshold)
                    if new_member is None:
                        break
                    group.append(new_member)

                dissimilar_member = DivergentGroupsGenerator.select_user_for_divergent_group(group, similarity_matrix,
                                                                                              user_id_indexes,
                                                                                              sim_threshold=cfg.dissimilar_threshold)
                if dissimilar_member is not None:
                    group.append(dissimilar_member)
                if len(group) == group_size:
                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'similar_one_divergent',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )
            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list

# class CounterExampleGroupsGenerator(GroupsGenerator):

#     @staticmethod
#     def select_user_for_counter_example_group(group, sim_matrix, user_id_indexes, sim_threshold, lower_bound, upper_bound):
#         assert lower_bound and upper_bound

#         candidate_ids = set(user_id_indexes.tolist()).difference(set(group))
#         if len(candidate_ids) == 0:
#             return None
#         else:
#             #selection = random.sample(candidate_ids, 1)
#             bounded_candidates = [] # Candidates fitting inside [lower_bound, upper_bound]
#             for candidate in candidate_ids:
#                 tmp_group = group + [candidate]
#                 print(tmp_group)
#                 sim = GroupsGenerator.compute_average_similarity(tmp_group, user_id_indexes, sim_matrix)
#                 print(f'Avg similarity: {sim}, lower: {lower_bound}, upper: {upper_bound}')
#                 if sim >= lower_bound and sim <= upper_bound:
#                     bounded_candidates.append(candidate)
#             if not bounded_candidates:
#                 print("no bounded candidates")
#                 return None

#             selection = random.sample(bounded_candidates, 1)
#             return selection[0]


#     def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
#         assert len(group_sizes_to_create) == 2, group_sizes_to_create
#         groups_list = list()
#         largest_group_sims = []
#         print(f'Group sizes to create: {group_sizes_to_create[::-1]}')
#         lower_bound = None
#         upper_bound = None
#         for group_size in group_sizes_to_create[::-1]:
#             print(f'Generating [CounterExample, size={group_size}], group_number_to_create={group_number_to_create}')
#             groups_size_list = list()
#             while (len(groups_size_list) < group_number_to_create):
#                 group = random.sample(user_id_set, 1)
#                 while len(group) < group_size:
#                     if group_size == group_sizes_to_create[-1]:
#                         # For the largest group, maximize similarity
#                         print(f"Searching for group with maximal similarity and size = {group_size}")
#                         new_member = SimilarGroupsGenerator.select_user_for_sim_group(group, similarity_matrix, user_id_indexes, sim_threshold=cfg.similar_threshold)
#                     else:
#                         # For smaller groups, select users to fit inside bounds
#                         print(f"Searching for group that fits within the bounds and size = {group_size}")
#                         new_member = CounterExampleGroupsGenerator.select_user_for_counter_example_group(group, similarity_matrix,
#                                                                                   user_id_indexes,
#                                                                                   cfg.similar_threshold,
#                                                                                   lower_bound, upper_bound)
#                     if new_member is None:
#                         break
#                     group.append(new_member)
#                 if len(group) == group_size:
#                     groups_size_list.append(
#                         {
#                             "group_size": group_size,
#                             "group_similarity": 'similar',
#                             "group_members": group,
#                             "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
#                         }
#                     )

#                     print(f'grp size {group_size}, {group_sizes_to_create}')
#                     if group_size == group_sizes_to_create[-1]:
#                         largest_group_sims.append(groups_size_list[-1]['avg_similarity'])
                        
#                     print(f'Group size={group_size}, avg_similarity={largest_group_sims[-1]}')
                
#             lower_bound, upper_bound = min(largest_group_sims), max(largest_group_sims)
#             print(f'lower_bound={lower_bound}, upper_bound={upper_bound}')

#             groups_list.extend(groups_size_list)
#             print(len(groups_list))
#         return groups_list    

class CounterExampleGroupsGenerator(GroupsGenerator):

    @staticmethod
    def select_users_for_counter_example_group(group, sim_matrix, user_id_indexes, sim_threshold, lower_bound, upper_bound, n_select):
        assert lower_bound and upper_bound

        user_id_indexes_list = user_id_indexes.tolist()
        group_members = np.array([user_id_indexes_list.index(member) for member in group])
        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size

        indexes = np.where((mean_similarities >= lower_bound) & (mean_similarities <= upper_bound))[0].tolist()
        ids_to_select_from = [user_id_indexes[index] for index in indexes]


        candidate_ids = set(ids_to_select_from).difference(group)
        if len(candidate_ids) < n_select:
            return None
        else:
            selection = random.sample(candidate_ids, n_select)
            return selection


    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        import time
        start_time = time.perf_counter()
        
       
        groups_list = list()
        largest_group_sims = []
        print(f'Group sizes to create: {group_sizes_to_create[::-1]}')
        lower_bound = None
        upper_bound = None
        for group_size in group_sizes_to_create[::-1]:
            print(f'Generating [CounterExample, size={group_size}], group_number_to_create={group_number_to_create} total time = {time.perf_counter() - start_time}')
            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                print(f"Rnd generated: {len(groups_size_list)}/{group_number_to_create}")
                while len(group) < group_size:
                    if group_size == group_sizes_to_create[-1]:
                        # For the largest group, maximize similarity
                        # new_member = SimilarGroupsGenerator.select_user_for_sim_group(group, similarity_matrix, user_id_indexes, sim_threshold=cfg.similar_threshold)
                        print(f"Searching for group with maximal similarity and size = {group_size}")
                        new_member = SimilarGroupsGenerator.select_users_for_sim_group(group, similarity_matrix,
                                                                                  user_id_indexes, n_to_select=group_size,
                                                                                  sim_threshold=cfg.similar_threshold)
                    else:
                        # For smaller groups, select users to fit inside bounds
                        # start_time_2 = time.perf_counter()
                        # print("Above expensive")
                        print(f"Searching for group that fits within the bounds [{lower_bound}, {upper_bound}], and size = {group_size}")
                        new_member = CounterExampleGroupsGenerator.select_users_for_counter_example_group(group, similarity_matrix,
                                                                                  user_id_indexes,
                                                                                  cfg.similar_threshold,
                                                                                  lower_bound, upper_bound, n_select=group_size)
                        # print(f"Expensive function took: {time.perf_counter() - start_time_2}")
                    
                    if new_member:
                        group = new_member
                    break
                if len(group) == group_size:
                    print("OK")
                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'counter_example',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )

                    print(f'grp size {group_size}, {group_sizes_to_create}')
                    if group_size == group_sizes_to_create[-1]:
                        largest_group_sims.append(groups_size_list[-1]['avg_similarity'])
                        
                    print(f'Group size={group_size}, avg_similarity={largest_group_sims[-1]}')
                else:
                    print(f"Weird: {group}, {group_size}")


            lower_bound, upper_bound = min(largest_group_sims) * 0.9, max(largest_group_sims) * 1.1
            print(f'lower_bound={lower_bound}, upper_bound={upper_bound}')

            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list


class BalancedCounterExampleGroupsGenerator(GroupsGenerator):

    @staticmethod
    def select_users_for_counter_example_group(group, sim_matrix, user_id_indexes, sim_threshold, lower_bound, upper_bound, n_select):
        assert lower_bound and upper_bound

        user_id_indexes_list = user_id_indexes.tolist()
        group_members = np.array([user_id_indexes_list.index(member) for member in group])
        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size

        indexes = np.where((mean_similarities >= lower_bound) & (mean_similarities <= upper_bound))[0].tolist()
        ids_to_select_from = [user_id_indexes[index] for index in indexes]


        candidate_ids = set(ids_to_select_from).difference(group)
        if len(candidate_ids) < n_select:
            return None
        else:
            selection = random.sample(candidate_ids, n_select)
            return selection


    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        import time
        start_time = time.perf_counter()
        
       
        groups_list = list()
        largest_group_sims = []
        print(f'Group sizes to create: {group_sizes_to_create[::-1]}')
        lower_bound = None
        upper_bound = None
        for group_size in group_sizes_to_create[::-1]:
            print(f'Generating [BalancedCounterExample, size={group_size}], group_number_to_create={group_number_to_create} total time = {time.perf_counter() - start_time}')
            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                print(f"Rnd generated: {len(groups_size_list)}/{group_number_to_create}")
                while len(group) < group_size:
                    if group_size == group_sizes_to_create[-1]:
                        # For the largest group, maximize similarity
                        # new_member = SimilarGroupsGenerator.select_user_for_sim_group(group, similarity_matrix, user_id_indexes, sim_threshold=cfg.similar_threshold)
                        print(f"Searching for group with maximal similarity and size = {group_size}")
                        new_member = SimilarGroupsGenerator.select_users_for_sim_group(group, similarity_matrix,
                                                                                  user_id_indexes, n_to_select=group_size,
                                                                                  sim_threshold=cfg.similar_threshold)
                    else:
                        # For smaller groups, select users to fit inside bounds
                        # start_time_2 = time.perf_counter()
                        # print("Above expensive")
                        print(f"Searching for group that fits within the bounds and size = {group_size}")
                        new_member = BalancedCounterExampleGroupsGenerator.select_users_for_counter_example_group(group, similarity_matrix,
                                                                                  user_id_indexes,
                                                                                  cfg.similar_threshold,
                                                                                  lower_bound, upper_bound, n_select=group_size)
                        # print(f"Expensive function took: {time.perf_counter() - start_time_2}")
                    
                    if new_member:
                        group = new_member
                    break
                if len(group) == group_size:
                    print("OK")
                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'balanced_counter_example',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )

                    print(f'grp size {group_size}, {group_sizes_to_create}')
                    if group_size == group_sizes_to_create[-1]:
                        largest_group_sims.append(groups_size_list[-1]['avg_similarity'])
                        
                    print(f'Group size={group_size}, avg_similarity={largest_group_sims[-1]}')
                else:
                    print(f"Weird: {group}, {group_size}")


            lower_bound, upper_bound = sum(largest_group_sims) / len(largest_group_sims), max(largest_group_sims)
            print(f'lower_bound={lower_bound}, upper_bound={upper_bound}')

            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list

class IncreasingSimilarityGroupsGenerator(GroupsGenerator):

    @staticmethod
    def select_users_for_counter_example_group(group, sim_matrix, user_id_indexes, sim_threshold, lower_bound, upper_bound, n_select):
        assert lower_bound and upper_bound

        user_id_indexes_list = user_id_indexes.tolist()
        group_members = np.array([user_id_indexes_list.index(member) for member in group])
        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size

        indexes = np.where((mean_similarities >= lower_bound) & (mean_similarities <= upper_bound))[0].tolist()
        ids_to_select_from = [user_id_indexes[index] for index in indexes]


        candidate_ids = set(ids_to_select_from).difference(group)
        if len(candidate_ids) < n_select:
            return None
        else:
            selection = random.sample(candidate_ids, n_select)
            return selection


    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        import time
        start_time = time.perf_counter()
        
       
        groups_list = list()
        largest_group_sims = []
        print(f'Group sizes to create: {group_sizes_to_create[::-1]}')
        lower_bound = None
        upper_bound = None
        for group_size in group_sizes_to_create[::-1]:
            print(f'Generating [IncreasingSimilarity, size={group_size}], group_number_to_create={group_number_to_create} total time = {time.perf_counter() - start_time}')
            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                print(f"Rnd generated: {len(groups_size_list)}/{group_number_to_create}")
                while len(group) < group_size:
                    if group_size == group_sizes_to_create[-1]:
                        # For the largest group, maximize similarity
                        # new_member = SimilarGroupsGenerator.select_user_for_sim_group(group, similarity_matrix, user_id_indexes, sim_threshold=cfg.similar_threshold)
                        print(f"Searching for group with maximal similarity and size = {group_size}")
                        new_member = SimilarGroupsGenerator.select_users_for_sim_group(group, similarity_matrix,
                                                                                  user_id_indexes, n_to_select=group_size,
                                                                                  sim_threshold=cfg.similar_threshold)
                    else:
                        # For smaller groups, select users to fit inside bounds
                        # start_time_2 = time.perf_counter()
                        # print("Above expensive")
                        print(f"Searching for group that fits within the bounds and size = {group_size}")
                        new_member = IncreasingSimilarityGroupsGenerator.select_users_for_counter_example_group(group, similarity_matrix,
                                                                                  user_id_indexes,
                                                                                  cfg.similar_threshold,
                                                                                  lower_bound, upper_bound, n_select=group_size)
                        # print(f"Expensive function took: {time.perf_counter() - start_time_2}")
                    
                    if new_member:
                        group = new_member
                    break
                if len(group) == group_size:
                    print("OK")
                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'increasing_similarity',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )

                    print(f'grp size {group_size}, {group_sizes_to_create}')
                    if group_size == group_sizes_to_create[-1]:
                        largest_group_sims.append(groups_size_list[-1]['avg_similarity'])
                        
                    print(f'Group size={group_size}, avg_similarity={largest_group_sims[-1]}')
                else:
                    print(f"Weird: {group}, {group_size}")


            lower_bound, upper_bound = min(largest_group_sims) / 2, min(largest_group_sims)
            print(f'lower_bound={lower_bound}, upper_bound={upper_bound}')

            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list


class Outliers(GroupsGenerator):

    @staticmethod
    def get_num_outliers(group_size):
        return max(int(round(group_size * 0.1)), 1)

    @staticmethod
    def select_users_for_counter_example_group(group, sim_matrix, user_id_indexes, sim_threshold, lower_bound, upper_bound, n_select):
        assert lower_bound and upper_bound

        user_id_indexes_list = user_id_indexes.tolist()
        group_members = np.array([user_id_indexes_list.index(member) for member in group])
        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size

        indexes = np.where((mean_similarities >= lower_bound) & (mean_similarities <= upper_bound))[0].tolist()
        ids_to_select_from = [user_id_indexes[index] for index in indexes]


        candidate_ids = set(ids_to_select_from).difference(group)
        if len(candidate_ids) < n_select:
            return None
        else:
            selection = random.sample(candidate_ids, n_select)
            return selection

    @staticmethod
    def find_outlier_centered_group(similar_group, n, sim_matrix, user_id_indexes):
        #ids_to_select_from = set(user_id_indexes) # in original version, this was: set()
        
        user_id_indexes_list = user_id_indexes.tolist()

        group_members = np.array([user_id_indexes_list.index(member) for member in similar_group])

        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size
        mean_similarities[group_members] = np.inf # Mask out those already in the group as they would have similarity = 0 and could be selected
        index = np.argmin(mean_similarities)
        
        outlier_similarities = sim_matrix[index:index+1].sum(axis=0)
        outlier_similarities[group_members] = 0.0
        outlier_similarities[index] = 0.0

        sorted_sim_indices = np.argsort(-outlier_similarities)

        new_group = [index] + sorted_sim_indices[:n-1].tolist()
        new_group = [user_id_indexes[index] for index in new_group]
        
        for u_id in new_group:
            assert u_id not in similar_group
        
        return new_group

    @staticmethod
    def find_outlier_diverged_group(similar_group, n, sim_matrix, user_id_indexes):
        #ids_to_select_from = set(user_id_indexes) # in original version, this was: set()
        
        user_id_indexes_list = user_id_indexes.tolist()

        group_members = np.array([user_id_indexes_list.index(member) for member in similar_group])

        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size
        mean_similarities[group_members] = np.inf # Mask out those already in the group as they would have similarity = 0 and could be selected
        index = np.argmin(mean_similarities)

        new_group = [index]

        for i in range(n - 1):
            new_group_arr = np.concatenate([group_members, np.array(new_group)])
            outlier_similarities = sim_matrix[new_group_arr].sum(axis=0) / new_group_arr.size
            outlier_similarities[new_group_arr] = np.inf
        
            new_group.append(np.argmin(outlier_similarities))

        #sorted_sim_indices = np.argsort(-outlier_similarities)

        #new_group = [index] + sorted_sim_indices[:n-1].tolist()
        
        
        new_group = [user_id_indexes[index] for index in new_group]
        
        for u_id in new_group:
            assert u_id not in similar_group
        
        return new_group

    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        import time
        groups_list = list()
        print(f'Group sizes to create: {group_sizes_to_create[::-1]}')
        for group_size in group_sizes_to_create[::-1]:

            num_outliers = Outliers.get_num_outliers(group_size)
            assert num_outliers <= group_size, f"num_outliers={num_outliers}, must be smaller than group_size={group_size}"
            similar_group_size = group_size - num_outliers

            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                group = random.sample(user_id_set, 1)
                while len(group) < similar_group_size:
                    print(f"Searching for group with maximal similarity and size = {similar_group_size}, sim_thresh = {cfg.similar_threshold}")
                    new_member = SimilarGroupsGenerator.select_users_for_sim_group(group, similarity_matrix,
                                                                                  user_id_indexes, n_to_select=similar_group_size,
                                                                                  sim_threshold=cfg.similar_threshold)

                    if new_member:
                        group = new_member
                    break
                if len(group) == similar_group_size:
                    print("OK, extend with outliers")
                    # Extend with outliers
                    # Find first, most divergning outlier and members most similar to it
                    # (we should have enough randomnes because the "similar" subgroup with similar_group_size was seeded by random member)
                    outlier_group = Outliers.find_outlier_centered_group(group, num_outliers, similarity_matrix, user_id_indexes)
                    
                    group = group + outlier_group



                    groups_size_list.append(
                        {
                            "group_size": group_size,
                            "group_similarity": 'outliers',
                            "group_members": group,
                            "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                        }
                    )

                else:
                    print(f"Weird: {group}, {group_size}")



            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list


class TwoSubGroupsGenerator(GroupsGenerator):

    @staticmethod
    def select_users_for_counter_example_group(group, sim_matrix, user_id_indexes, sim_threshold, lower_bound, upper_bound, n_select):
        assert lower_bound and upper_bound

        user_id_indexes_list = user_id_indexes.tolist()
        group_members = np.array([user_id_indexes_list.index(member) for member in group])
        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size

        indexes = np.where((mean_similarities >= lower_bound) & (mean_similarities <= upper_bound))[0].tolist()
        ids_to_select_from = [user_id_indexes[index] for index in indexes]


        candidate_ids = set(ids_to_select_from).difference(group)
        if len(candidate_ids) < n_select:
            return None
        else:
            selection = random.sample(candidate_ids, n_select)
            return selection

    @staticmethod
    def find_outlier_centered_group(similar_group, n, sim_matrix, user_id_indexes):
        #ids_to_select_from = set(user_id_indexes) # in original version, this was: set()
        user_id_indexes_list = user_id_indexes.tolist()

        group_members = np.array([user_id_indexes_list.index(member) for member in similar_group])

        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size
        mean_similarities[group_members] = np.inf # Mask out those already in the group as they would have similarity = 0 and could be selected
        index = np.argmin(mean_similarities)
        
        outlier_similarities = sim_matrix[index:index+1].sum(axis=0)
        outlier_similarities[group_members] = 0.0
        outlier_similarities[index] = 0.0

        sorted_sim_indices = np.argsort(-outlier_similarities)

        new_group = [index] + sorted_sim_indices[:n-1].tolist()
        new_group = [user_id_indexes[index] for index in new_group]
        
        for u_id in new_group:
            assert u_id not in similar_group
        
        return new_group

    @staticmethod
    def find_similar_neighbors(member, n, sim_matrix, user_id_indexes):

        user_id_indexes_list = user_id_indexes.tolist()

        group_members = np.array([user_id_indexes_list.index(member)])

        mean_similarities = sim_matrix[group_members].sum(axis=0) / group_members.size
        mean_similarities[group_members] = 0
        
        sorted_sim_indices = np.argsort(-mean_similarities)
        
        new_group = [group_members[0]] + sorted_sim_indices[:n].tolist()
        new_group = [user_id_indexes[index] for index in new_group]
        
        
        return new_group

    def generateGroups(self, user_id_indexes, user_id_set, similarity_matrix, group_sizes_to_create, group_number_to_create):
        import time
        groups_list = list()
        print(f'Group sizes to create: {group_sizes_to_create[::-1]}')
        for group_size in group_sizes_to_create[::-1]:

            assert group_size % 2 == 0, f"Group size ({group_size}) must be divisible by 2"
            subgroup_size = group_size // 2

            groups_size_list = list()
            while (len(groups_size_list) < group_number_to_create):
                sub_group = random.sample(user_id_set, 1)
                sub_group = TwoSubGroupsGenerator.find_similar_neighbors(sub_group[0], subgroup_size - 1, similarity_matrix, user_id_indexes)
                second_sub_group = TwoSubGroupsGenerator.find_outlier_centered_group(sub_group, subgroup_size, similarity_matrix, user_id_indexes)
                group = sub_group + second_sub_group
                assert len(group) == group_size
                assert len(sub_group) == subgroup_size
                assert len(second_sub_group) == subgroup_size
                print(f"Generated subgroups")
                
                groups_size_list.append(
                    {
                        "group_size": group_size,
                        "group_similarity": 'two_sub_groups',
                        "group_members": group,
                        "avg_similarity": GroupsGenerator.compute_average_similarity(group, user_id_indexes, similarity_matrix)
                    }
                )



            groups_list.extend(groups_size_list)
            print(len(groups_list))
        return groups_list