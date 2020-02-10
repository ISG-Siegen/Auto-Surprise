
class BasicReduction(object):
    """
    A basic strategy for comparison of algorithms
    """
    
    def filter_algorithms(self, tasks, algorithms):
        """
        Rank N algorithms and take the top N/2 algorithms which performed
        better than baseline result for the next iteration
        """
        filtered_algorithms = dict(filter(lambda algo: algo[1]['above_baseline'], tasks.items()))
        algorithms_ranking = [i[0] for i in sorted(filtered_algorithms.items(), key=lambda x: x[1]['score']['loss'], reverse=False)]
        algorithms_count = round(len(algorithms) / 2)
        algorithms = algorithms_ranking[0:algorithms_count]

        return algorithms
