use strict;
use warnings;
use File::Find;
# use Text::Diff::Parser;
use File::Temp;

# parameters for controlling what gets reported
my $max_line_len = 121;
my $overlength_threshold = 25;

# process all the lines in the file and check for tabs, eof newline, overlength conditions, etc.
sub process_file_lines
    {
    my $fname = $_[0];
    # for checking the final newline
    my $last_line;
    
    # open the file
    open(FILE, "< $fname") or die "can't open $fname: $!";
    
    # initialize counters to 0
    my $tab_count = 0;
    my $line_count = 0;
    my $overlength_count = 0;
    # loop through all lines in the file and add up counters
    while (<FILE>)
        {
        $tab_count += tr/\t//;
        $last_line = $_ if eof;

        if (length($_) > $max_line_len)
            {
            $overlength_count += 1;
            }

        $line_count += 1;
        }
    close(FILE);

    my $message = "";

    if ($tab_count > 0)
        {
        $message .= "tabs:                $tab_count\n";
        }
    if ($overlength_count > $overlength_threshold)
        {
        $message .= "lines overlength:    $overlength_count\n";
        }
#    if (not $last_line =~ /^\n/)
#        {
#        $message .= "end of file newline: missing\n";
#        }

    return ($message, $line_count);
    }

sub wanted 
    {
    my $fname = $_;    
    
    # skip processing if this file is in the extern directory
    if ($File::Find::name =~ /\/extern\//)
        {
        return;
        }
    
    # skip processing if this file is in the share directory
    if ($File::Find::name =~ /\/share\//)
        {
        return;
        }
    
    if (/\.cc$/ or /\.h$/ or /\.cu$/ or /\.cuh$/ or /\.py$/)
        {
        my $full_message = "";
        my $message;
        my $line_count;
        ($message, $line_count) = process_file_lines($fname);
        $full_message .= $message;

        if ($full_message)
            {
            print "$File::Find::name\n";
            print $full_message;
            print "\n";
            }
        }
    }

# grep through the source and look for problems
finddepth(\&wanted, '.');
