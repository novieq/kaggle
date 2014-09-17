#!/usr/bin/perl
#use strict;

my $usage = "$0 <train.csv>";
my $fileName = shift @ARGV || die $usage;
open(INFILE, "$fileName") || die "Cannot open $fileName";
my $line = <INFILE>;
#Id,Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26
my @aoh; #Array of hashes. Array location 0 is for the hash containing the categorical values of C1. 
my @aohClicks;

while($line = <INFILE>) {
	my @tokens = split(',', $line);
	foreach my $i (1..26) { #$i loops from C1 to C26 iterating over the respective hashes
		chomp $tokens[14+$i];
		$aoh[$i]{$tokens[14+$i]}=$aoh[$i]{$tokens[14+$i]}+1; #Adding impressions for every categorical value of Ci
		$aohClicks[$i]{$tokens[14+$i]}=$aohClicks[$i]{$tokens[14+$i]}+$tokens[1];
	}
}

foreach my $i (1..26) {
	while (my ($k,$v)=each $aoh[$i]){print "C${i},$k,$v,$aohClicks[$i]{$k},",$aohClicks[$i]{$k}/$v,"\n"};
}
